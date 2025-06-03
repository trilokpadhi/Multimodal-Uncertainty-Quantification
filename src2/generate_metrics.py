import os
import time
import pickle
import math
import itertools
import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import nltk
nltk.download('punkt')
from rouge_score import rouge_scorer

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from concurrent.futures import ProcessPoolExecutor
from multiprocessing import Manager

# For progress bar
from tqdm import tqdm

##############################################################################
# UTILITY
##############################################################################

def make_dir_if_not_exists(path):
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)

def min_max_scale(arr):
    arr = np.array(arr, dtype=float)
    mn, mx = arr.min(), arr.max()
    rng = mx - mn
    if rng < 1e-9:
        return [1.0]*len(arr)
    return (arr - mn)/rng

def plot_reliability_diagram(confidences, accuracies, label_str, out_path):
    bins = np.linspace(0, 1, 11)
    bin_idx = np.digitize(confidences, bins) - 1

    avg_accs = []
    centers = []
    for i in range(len(bins)-1):
        mask = (bin_idx == i)
        if mask.any():
            avg_acc = np.mean(accuracies[mask])
            c = (bins[i] + bins[i+1]) / 2
            avg_accs.append(avg_acc)
            centers.append(c)
    avg_accs = np.array(avg_accs)
    centers  = np.array(centers)

    plt.figure(figsize=(6, 5))
    plt.plot(centers, avg_accs, 'o-', label=label_str)
    plt.plot([0,1],[0,1],'--', color='gray', label='Perfect')
    plt.fill_between(centers, centers, avg_accs, alpha=0.1)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(f"Reliability Diagram: {label_str}")
    plt.legend(loc='lower right')

    if len(centers) > 0:
        ece = np.mean(np.abs(avg_accs - centers))
        mce = np.max(np.abs(avg_accs - centers))
    else:
        ece, mce = 0, 0
    txt = f"ECE: {ece:.3f}\nMCE: {mce:.3f}"
    plt.text(0.05, 0.95, txt, transform=plt.gca().transAxes,
             verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    make_dir_if_not_exists(os.path.dirname(out_path))
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()
    print(f"[INFO] Saved reliability diagram => {out_path}")

def fit_polynomial_regression(X, y, degree=2, alpha=3.0):
    pipe = make_pipeline(
        PolynomialFeatures(degree=degree),
        Ridge(alpha=alpha)
    )
    pipe.fit(X, y)
    return pipe

def apply_poly_calibration(pipe, X):
    raw = pipe.predict(X)
    return min_max_scale(raw)

def initialize_rouge():
    return rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

def chunkify(lst, chunk_size=16):
    for i in range(0, len(lst), chunk_size):
        yield lst[i:i+chunk_size]

##############################################################################
# BASELINE METRICS HELPERS
##############################################################################

def calculate_predictive_entropy(log_probs):
    probs = np.exp(log_probs)
    probs /= (probs.sum() + 1e-9)
    return float(- np.sum(probs * np.log(probs + 1e-9)))

def get_lexical_similarity(responses, rouge_obj):
    if len(responses) < 2:
        return 1.0
    pairs = itertools.combinations(responses, 2)
    scores = []
    for (r1, r2) in pairs:
        s = rouge_obj.score(r1, r2)['rougeL'].fmeasure
        scores.append(s)
    return float(np.mean(scores)) if scores else 0.0

def get_semantic_entropy(question, resp_to_logp, tokenizer, model, device='cpu'):
    """
    Batched approach for mutual entailment among responses.
    """
    exps = list(resp_to_logp.keys())
    logps= np.array(list(resp_to_logp.values()), dtype=float)
    pvals= np.exp(logps)
    pvals /= (pvals.sum() + 1e-9)

    n = len(exps)
    if n < 2:
        # 0 or 1 response => trivial
        return 0.0, n

    pairs = list(itertools.combinations(range(n), 2))
    if not pairs:
        return 0.0, n

    fwd_inps = []
    bwd_inps = []
    fwd_map = {}
    bwd_map = {}

    for idx, (i, j) in enumerate(pairs):
        premise_f = f"{question} {exps[i]}"
        hyp_f     = f"{question} {exps[j]}"
        fwd_inps.append( (i,j, premise_f, hyp_f) )
        fwd_map[(i,j)] = idx

        premise_b = f"{question} {exps[j]}"
        hyp_b     = f"{question} {exps[i]}"
        bwd_inps.append( (j,i, premise_b, hyp_b) )
        bwd_map[(j,i)] = idx

    # Forward pass
    fwd_logits = []
    for chunk in chunkify(fwd_inps, 16):
        premises = [c[2] for c in chunk]
        hyps     = [c[3] for c in chunk]
        inp = tokenizer(premises, hyps, return_tensors='pt', truncation=True, padding=True)
        inp = {k:v.to(device) for k,v in inp.items()}
        with torch.no_grad():
            out = model(**inp).logits
        fwd_logits.append(out.cpu())
    fwd_logits = torch.cat(fwd_logits, dim=0) if fwd_logits else torch.empty(0,3)

    # Backward pass
    bwd_logits = []
    for chunk in chunkify(bwd_inps, 16):
        premises = [c[2] for c in chunk]
        hyps     = [c[3] for c in chunk]
        inp = tokenizer(premises, hyps, return_tensors='pt', truncation=True, padding=True)
        inp = {k:v.to(device) for k,v in inp.items()}
        with torch.no_grad():
            out = model(**inp).logits
        bwd_logits.append(out.cpu())
    bwd_logits = torch.cat(bwd_logits, dim=0) if bwd_logits else torch.empty(0,3)

    # adjacency for mutual entailment
    mutual = [[False]*n for _ in range(n)]
    for (i,j) in pairs:
        idx_f = fwd_map[(i,j)]
        idx_b = bwd_map[(j,i)]
        pred_f = torch.argmax(fwd_logits[idx_f]).item() if idx_f < len(fwd_logits) else 0
        pred_b = torch.argmax(bwd_logits[idx_b]).item() if idx_b < len(bwd_logits) else 0
        if pred_f == 2 and pred_b == 2:  # label=2 => entailment
            mutual[i][j] = True
            mutual[j][i] = True

    # cluster with BFS/DFS 
    visited = [False]*n
    clusters = []
    for i in range(n):
        if visited[i]:
            continue
        stack = [i]
        visited[i] = True
        c = []
        while stack:
            node = stack.pop()
            c.append(node)
            for nb in range(n):
                if mutual[node][nb] and not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(c)

    cluster_probs = []
    for c in clusters:
        cluster_probs.append(np.mean(pvals[c]))

    cluster_probs = np.array(cluster_probs)
    cluster_probs /= (cluster_probs.sum() + 1e-9)
    sem_ent = - np.sum(cluster_probs * np.log(cluster_probs + 1e-9))
    return float(sem_ent), len(clusters)

##############################################################################
# MULTI-GPU: DISTRIBUTE FILES
##############################################################################

def distribute_files_across_gpus(files, num_gpus):
    if num_gpus < 1:
        # fallback: single CPU
        return [(0, files)]
    chunk_size = math.ceil(len(files) / num_gpus)
    pairs = []
    start = 0
    for gpu_id in range(num_gpus):
        subset = files[start:start+chunk_size]
        if subset:
            pairs.append((gpu_id, subset))
        start += chunk_size
    return pairs

##############################################################################
# STEP 1 (BASELINE) WITH GRANULAR PROGRESS
##############################################################################

def worker_baseline(files_subset, gpu_id, explanation_dir, progress_queue):
    """
    Each worker:
      - Loads DeBERTa on GPU gpu_id
      - Processes each file
      - After each file => puts a message into progress_queue
      - Returns list of results
    """
    device_str = f'cuda:{gpu_id}' if (torch.cuda.is_available() and gpu_id < torch.cuda.device_count()) else 'cpu'
    print(f"[Worker-Baseline] GPU {gpu_id} => {len(files_subset)} files. Device: {device_str}")

    deberta_name='microsoft/deberta-base-mnli'
    tokenizer = AutoTokenizer.from_pretrained(deberta_name)
    model = AutoModelForSequenceClassification.from_pretrained(deberta_name).to(device_str)
    model.eval()

    rouge_obj = initialize_rouge()
    results = []

    for file in files_subset:
        path = os.path.join(explanation_dir, file)
        with open(path,'rb') as f:
            data = pickle.load(f)

        question_id = file.split('_')[1].split('.')[0]
        question = data['questions'][0]

        all_log_probs = []
        all_responses = []
        resp_to_logp  = {}

        for i in range(20):
            key = f"response_{i}"
            if key not in data:
                continue
            sub = data[key]
            tscores = sub.get('transition_scores',[])
            tscores = [t for t in tscores if np.isfinite(t).all()]
            logprob = float(np.sum(tscores)) if len(tscores) > 0 else float('nan')
            resp_txt= sub.get('decoded_outputs','')
            all_log_probs.append(logprob)
            all_responses.append(resp_txt)
            resp_to_logp[resp_txt] = logprob

        clean_log_probs= [lp for lp in all_log_probs if not math.isnan(lp)]
        if not clean_log_probs:
            results.append({
                'question_id': question_id,
                'predictive_entropy': 0.0,
                'lexical_similarity': 0.0,
                'semantic_entropy': 0.0,
                'num_clusters': 0,
                'responses': all_responses,
                'resp_log_probs': all_log_probs,
                'question': question
            })
        else:
            # compute metrics
            pe = calculate_predictive_entropy(np.array(clean_log_probs))
            ls = get_lexical_similarity(all_responses, rouge_obj)
            sem_ent, n_clus = get_semantic_entropy(question, resp_to_logp, tokenizer, model, device_str)

            results.append({
                'question_id': question_id,
                'predictive_entropy': pe,
                'lexical_similarity': ls,
                'semantic_entropy': sem_ent,
                'num_clusters': n_clus,
                'responses': all_responses,
                'resp_log_probs': all_log_probs,
                'question': question
            })

        # Signal that a file has been processed
        progress_queue.put(1)

    return results

def get_baseline_results(explanation_dir, out_csv, max_files=None):
    """
    Multi-GPU Baseline with a single global tqdm that increments per file.
    """
    if os.path.isfile(out_csv):
        print(f"[INFO] Baseline results already exist => {out_csv}")
        return pd.read_csv(out_csv)

    print(f"[INFO] Step 1 (Baseline): Gathering .pkl files from {explanation_dir}...")
    all_files = sorted([f for f in os.listdir(explanation_dir) if f.endswith('.pkl')])
    if max_files is not None:
        all_files = all_files[:max_files]
    total_count = len(all_files)

    if total_count == 0:
        print(f"[WARNING] No .pkl files found in {explanation_dir}")
        pd.DataFrame([]).to_csv(out_csv, index=False)
        return pd.DataFrame([])

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        num_gpus = 1

    print(f"[INFO] Found {len(all_files)} files. Using {num_gpus} GPU(s).")
    dist = distribute_files_across_gpus(all_files, num_gpus)

    # Create a manager and a progress queue
    manager = Manager()
    progress_queue = manager.Queue()

    # Start parallel processing
    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=len(dist)) as executor:
        for (gpu_id, subset) in dist:
            fut = executor.submit(worker_baseline, subset, gpu_id, explanation_dir, progress_queue)
            futures.append(fut)

        # Show a single global bar in the main process
        with tqdm(total=total_count, desc="Step1-Baseline") as pbar:
            processed = 0
            while processed < total_count:
                # Check the queue for updates
                while not progress_queue.empty():
                    progress_queue.get()
                    processed += 1
                    pbar.update(1)
                time.sleep(0.1)  # Adjust sleep to balance responsiveness and CPU usage
            # Process any remaining updates after loop exit
            while not progress_queue.empty():
                progress_queue.get()
                processed += 1
                pbar.update(1)

    # Now collect the final results from each future
    for fut in futures:
        worker_result = fut.result()
        results.extend(worker_result)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Baseline results => {out_csv}")
    return df

##############################################################################
# STEP 2 (ACCURACY) WITH GRANULAR PROGRESS
##############################################################################

def worker_accuracy(files_subset, gpu_id, explanation_dir, progress_queue, add_question_for_accuracy_calculation):
    """
    Each worker:
      - Loads DeBERTa on GPU gpu_id
      - For each file => computes accuracy
      - Puts a message into progress_queue after each file
      - Returns results
    """
    device_str = f'cuda:{gpu_id}' if (torch.cuda.is_available() and gpu_id < torch.cuda.device_count()) else 'cpu'
    print(f"[Worker-Accuracy] GPU {gpu_id} => {len(files_subset)} files. Device: {device_str}")

    deberta_name='microsoft/deberta-base-mnli'
    tokenizer = AutoTokenizer.from_pretrained(deberta_name)
    model = AutoModelForSequenceClassification.from_pretrained(deberta_name).to(device_str)
    model.eval()

    results = []

    for file in files_subset:
        path = os.path.join(explanation_dir, file)
        with open(path,'rb') as f:
            data= pickle.load(f)

        question_id = file.split('_')[1].split('.')[0]
        question = data['questions'][0]
        gt_answer= data['answers'][0]

        responses=[]
        for i in range(20):
            key=f"response_{i}"
            if key not in data:
                continue
            sub=data[key]
            resp_txt= sub.get('decoded_outputs',"")
            responses.append(resp_txt)

        if not responses:
            accuracy = 0.0
        else:
            # Batched inference
            preds = []
            for chunk in chunkify(responses, 16):
                if add_question_for_accuracy_calculation:
                    premises = [f"{question} {resp}" for resp in chunk]
                    hyps     = [f"{question} {gt_answer}"]*len(chunk)
                else:
                    premises = chunk
                    hyps     = [gt_answer]*len(chunk)
                
                inp = tokenizer(premises, hyps, return_tensors='pt', truncation=True, padding=True)
                inp = {k:v.to(device_str) for k,v in inp.items()}
                with torch.no_grad():
                    lg = model(**inp).logits
                chunk_preds = torch.argmax(lg, dim=1).tolist()
                preds.extend(chunk_preds)

            correct = sum([1 for p in preds if p==2])  # label=2 => entailment
            accuracy = correct / len(responses)

        results.append({'question_id': question_id, 'accuracy': accuracy})

        # Signal that a file has been processed
        progress_queue.put(1)

    return results

def get_accuracy(explanation_dir, out_csv, max_files=None, add_question_for_accuracy_calculation=False):
    if os.path.isfile(out_csv):
        print(f"[INFO] Accuracy results already exist => {out_csv}")
        return pd.read_csv(out_csv)

    print(f"[INFO] Step 2 (Accuracy): Gathering .pkl files from {explanation_dir}...")
    all_files = sorted([f for f in os.listdir(explanation_dir) if f.endswith('.pkl')])
    if max_files is not None:
        all_files = all_files[:max_files]
    total_count = len(all_files)

    if total_count == 0:
        print(f"[WARNING] No .pkl files found in {explanation_dir}")
        pd.DataFrame([]).to_csv(out_csv, index=False)
        return pd.DataFrame([])

    num_gpus = torch.cuda.device_count()
    if num_gpus < 1:
        num_gpus = 1

    print(f"[INFO] Found {total_count} files. Using {num_gpus} GPU(s).")
    dist = distribute_files_across_gpus(all_files, num_gpus)

    # Create a manager and a progress queue
    manager = Manager()
    progress_queue = manager.Queue()

    futures = []
    results = []
    with ProcessPoolExecutor(max_workers=len(dist)) as executor:
        for (gpu_id, subset) in dist:
            fut = executor.submit(worker_accuracy, subset, gpu_id, explanation_dir, progress_queue, add_question_for_accuracy_calculation)
            futures.append(fut)

        # Single global progress bar
        with tqdm(total=total_count, desc="Step2-Accuracy") as pbar:
            processed = 0
            while processed < total_count:
                while not progress_queue.empty():
                    progress_queue.get()
                    processed += 1
                    pbar.update(1)
                time.sleep(0.1)
            # Process any remaining updates
            while not progress_queue.empty():
                progress_queue.get()
                processed += 1
                pbar.update(1)

    # gather results
    for fut in futures:
        worker_result = fut.result()
        results.extend(worker_result)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Accuracy => {out_csv}")
    return df

##############################################################################
# STEP 3: GET GROUNDING
##############################################################################

def load_grounding_dict(grounding_folder, score_key='biomedclip_score', score_type='binary'):
    out = {}
    out_processed = {}
    if not os.path.isdir(grounding_folder):
        print(f"[WARNING] Missing folder: {grounding_folder}")
        return out, out_processed

    pkls = [p for p in os.listdir(grounding_folder) if p.endswith('.pkl')]
    for pklf in pkls:
        fpath = os.path.join(grounding_folder, pklf)
        if os.path.getsize(fpath) == 0:
            continue
        with open(fpath, 'rb') as f:
            data = pickle.load(f)
        # qid = int(data.get('question_ids', None)[0])
        # print(data)
        try:
            qid = int(data.get('qids', None)[0])
        except (TypeError, ValueError, IndexError, KeyError):
            try:
                qid = int(data.get('question_id', None))
            except (TypeError, ValueError, IndexError, KeyError):
                try:
                    qid = int(data.get('question_ids', None)[0])
                except (TypeError, ValueError, IndexError, KeyError) as e:
                    print(f"[ERROR] Could not find question_id in {fpath}: {e}")
                    exit()
        # qid = int(data.get('question_id', None))
        val_list = []
        for key in data.keys():
            if 'response' in key:
                val_list.append(data[key].get(score_key, "No"))

        if qid:
            out[qid] = val_list
            if score_type == 'binary':
                temp = ['yes' if "yes" in v.lower() else 'no' for v in val_list]
                out_processed[qid] = temp
            elif score_type == 'continuous':
                temp = [float(v) if isinstance(v, (int, float)) else 0.0 for v in val_list]
                out_processed[qid] = temp
            else:
                print(f"[WARNING] Unknown score_type: {score_type}")
                out_processed[qid] = val_list
    return out, out_processed

# def get_grounding(llama32_11b_dir, qwen_vl_dir, qwen_vl_25_dir, out_csv):
#     """
#     Grounding is typically quick, so we do it in a single process (CPU).
#     """
#     if os.path.isfile(out_csv):
#         print(f"[INFO] Grounding results already exist => {out_csv}")
#         return pd.read_csv(out_csv)

#     # dict_biomedclip, dict_biomedclip_processed   = load_grounding_dict(biomedclip_dir,   score_key='biomedclip_score')
#     dict_llama32_11b, dict_llama32_11b_processed  = load_grounding_dict(llama32_11b_dir,  score_key='llama_32_response')
#     # dict_llama32_70b, dict_llama32_70b_processed  = load_grounding_dict(llama32_70b_dir,  score_key='llama_32_response')
#     dict_qwen_vl, dict_qwen_vl_processed    = load_grounding_dict(qwen_vl_dir, score_key='qwen_vl_response')
#     dict_qwen_vl_25, dict_qwen_vl_25_processed = load_grounding_dict(qwen_vl_25_dir, score_key='qwen_vl_response')
    

#     # all_qids= set(dict_biomedclip.keys()).union(dict_llama32_11b.keys(), dict_llama32_70b.keys(), dict_qwen_vl.keys())
#     all_qids= set(dict_llama32_11b.keys()).union(dict_qwen_vl.keys(), dict_qwen_vl_25.keys())
    
#     results=[]
#     for qid in sorted(all_qids, key=lambda x:int(x)):
#         # s_bio= dict_biomedclip.get(qid, None)
#         # s_bio_processed= dict_biomedclip_processed.get(qid, None)
#         s_11b= dict_llama32_11b.get(qid, None) 
#         s_11b_processed= dict_llama32_11b_processed.get(qid, None)
#         # s_70b= dict_llama32_70b.get(qid, None)
#         # s_70b_processed= dict_llama32_70b_processed.get(qid, None)
#         s_qwen= dict_qwen_vl.get(qid, None)
#         s_qwen_processed= dict_qwen_vl_processed.get(qid, None)
#         s_qwen_25= dict_qwen_vl_25.get(qid, None)
#         s_qwen_25_processed= dict_qwen_vl_25_processed.get(qid, None)
#         rec={
#             'question_id': qid,
#             # 'grounding_biomedclip': s_bio,
#             # 'grounding_biomedclip_processed': s_bio_processed,
#             'grounding_llama32_11b': s_11b,
#             'grounding_llama32_11b_processed': s_11b_processed,
#             # 'grounding_llama32_70b': s_70b,
#             # 'grounding_llama32_70b_processed': s_70b_processed,
#             'grounding_qwen_vl': s_qwen,
#             'grounding_qwen_vl_processed': s_qwen_processed,
#             'grounding_qwen_vl_25': s_qwen_25,
#             'grounding_qwen_vl_25_processed': s_qwen_25_processed
#         }
#         results.append(rec)
#     df = pd.DataFrame(results)
#     df.to_csv(out_csv, index=False)
#     print(f"[INFO] Grounding => {out_csv}")
#     return df

def get_grounding(out_csv, grounding_info):
    """
    Grounding is typically quick, so we do it in a single process (CPU).
    """
    if os.path.isfile(out_csv):
        print(f"[INFO] Grounding results already exist => {out_csv}")
        return pd.read_csv(out_csv)

    all_dicts = {}
    all_processed_dicts = {}

    for key, info in grounding_info.items():
        folder = info['folder']
        score_key = info['score_key']
        if 'biomedclip' in key:
            score_type = 'continuous'
            dict_data, dict_processed = load_grounding_dict(folder, score_key=score_key, score_type=score_type)
        else:
            dict_data, dict_processed = load_grounding_dict(folder, score_key=score_key)
        all_dicts[key] = dict_data
        all_processed_dicts[key] = dict_processed

    all_qids = set()
    for dict_data in all_dicts.values():
        all_qids.update(dict_data.keys())

    results = []
    for qid in sorted(all_qids, key=lambda x: int(x)):
        rec = {'question_id': qid}
        for key in grounding_info.keys():
            rec[f'grounding_{key}'] = all_dicts[key].get(qid, None)
            rec[f'grounding_{key}_processed'] = all_processed_dicts[key].get(qid, None)
        results.append(rec)

    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False)
    print(f"[INFO] Grounding => {out_csv}")
    return df

##############################################################################
# STEP 4: MERGE
##############################################################################

def merge_baseline_accuracy_grounding(baseline_csv, accuracy_csv, grounding_csv, out_csv):
    if os.path.isfile(out_csv):
        print(f"[INFO] Merged results already exist => {out_csv}")
        return pd.read_csv(out_csv)

    df_base= pd.read_csv(baseline_csv)
    df_acc = pd.read_csv(accuracy_csv)
    df_grd = pd.read_csv(grounding_csv)

    df_baseacc= pd.merge(df_base, df_acc, on='question_id', how='outer')
    df_final= pd.merge(df_baseacc, df_grd, on='question_id', how='outer')
    df_final.to_csv(out_csv, index=False)
    print(f"[INFO] Merged => {out_csv}")
    return df_final




def main():
    """
    Steps:
      1) Get Baseline results (predictive_entropy, lexical_similarity, semantic_entropy) [multi-GPU + granular progress]
      2) Get Accuracy [multi-GPU + granular progress]
      3) Get Grounding [CPU is fine]
      4) Merge
      5) Plots
    """
    explanation_dir = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/gemini_slake/explanations" # for slake
    # explanation_dir = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/explanations_2000" # for vqa
    # out_dir         = "my_outputs/vqa" # for vqa
    out_dir = "my_outputs_slake_gemini_filtered_june1" # for slake 
    make_dir_if_not_exists(out_dir)

    # Step 1: Baseline
    baseline_csv = os.path.join(out_dir, "baseline.csv")
    get_baseline_results(
        explanation_dir,
        baseline_csv,
        max_files=None  # or set an integer for quick testing
    )

    # Step 2: Accuracy
    accuracy_csv = os.path.join(out_dir, "accuracy.csv")
    get_accuracy(
        explanation_dir,
        accuracy_csv,
        max_files=None, 
        add_question_for_accuracy_calculation=True
    )

    # Step 3: Grounding (single process, typically quick)
    # grounding_biomedclip_folder = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding" 
    # grounding_llama32_11b_folder= "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32"
    # grounding_llama32_11b_folder = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/grounding_with_llama32_11B'
    # # grounding_llama32_70b_folder= "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32_90b"
    # grounding_qwen_vl_folder   = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/grounding_with_qwen_vl"
    # grounding_qwen_25_7B_vl_folder = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/grounding_with_qwen_vl_25_7B"
    # grounding_csv= os.path.join(out_dir, "grounding.csv")
    # grounding_gemini = "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_gemini"
    grounding_info = {
    'biomedclip': {
        'folder': '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/gemini_slake/grounding',
        'score_key': 'biomedclip_score',
        'score_type': 'continuous'
    },
    'llama32_11b': {
        # 'folder': '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_llama32',
        'folder': '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/gemini_slake/grounding_with_llama32_11B',
        'score_key': 'llama_32_response'
    },
    # # 'qwen_vl': {
    # #     'folder': "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_vqa6/llava_vqa_yes_gsam_grounding_random_1000_temp_05/grounding_with_qwen_vl",
    # #     'score_key': 'qwen_vl_response'
    # # },
    'qwen_vl': {
        # 'folder': "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_qwen_vl",
        'folder': '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/gemini_slake/grounding_with_qwen_vl',
        'score_key': 'qwen_vl_response'
    },
    # 'gemini': {
    #     'folder': "/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/runs_slake/llava_med_slake/grounding_with_gemini",
    #     'score_key': 'gemini_grounding_response'
    # }
    }
    grounding_csv = os.path.join(out_dir, "grounding.csv")
    get_grounding(grounding_csv, grounding_info)
    
    # get_grounding(
    #     # grounding_biomedclip_folder, 
    #     grounding_llama32_11b_folder,
    #     # grounding_llama32_70b_folder,
    #     grounding_qwen_25_7B_vl_folder,
    #     grounding_qwen_vl_folder,
    #     grounding_gemini,
    #     grounding_csv
    # )

    # Step 4: Merge
    merged_csv= os.path.join(out_dir, "merged.csv")
    merge_baseline_accuracy_grounding(baseline_csv, accuracy_csv, grounding_csv, merged_csv)


    print("\n[INFO] All steps complete!")

if __name__ == "__main__":
    main()

