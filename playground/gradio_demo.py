# # # # import gradio as gr
# # # # import pandas as pd
# # # # import matplotlib.pyplot as plt
# # # # import numpy as np

# # # # # Load your preprocessed DataFrame
# # # # df = pd.read_pickle('/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/high_consistency_low_acc_low_gr.csv')  # Assuming you saved the results

# # # # def show_question(index):
# # # #     row = df.iloc[index]
# # # #     responses = "\n\n".join([f"Response {i+1}: {resp}" for i, resp in enumerate(row.model_responses)])
    
# # # #     # Create consistency visualization
# # # #     fig, ax = plt.subplots(figsize=(6, 3))
# # # #     ax.bar(['Consistency', 'Grounding', 'Accuracy'], 
# # # #            [row.self_consistency_score, row.grounding_score, row.accuracy],
# # # #            color=['#4CAF50', '#2196F3', '#FF9800'])
# # # #     ax.set_ylim(0, 1)
# # # #     plt.tight_layout()
    
# # # #     return (
# # # #         row.question,
# # # #         row.answer,
# # # #         row.grounding_score,
# # # #         row.self_consistency_score,
# # # #         row.accuracy,
# # # #         responses,
# # # #         fig
# # # #     )

# # # # def build_demo():
# # # #     with gr.Blocks(theme=gr.themes.Soft(), title="Medical QA Analysis") as demo:
# # # #         gr.Markdown("# 🏥 Medical QA Evaluation Dashboard")
# # # #         gr.Markdown("Analyzing model performance on medical questions")
        
# # # #         with gr.Row():
# # # #             with gr.Column(scale=1):
# # # #                 index_slider = gr.Slider(0, len(df)-1, step=1, label="Question Index")
# # # #                 with gr.Row():
# # # #                     prev_btn = gr.Button("← Previous")
# # # #                     next_btn = gr.Button("Next →")
# # # #             with gr.Column(scale=2):
# # # #                 question = gr.Textbox(label="Question", interactive=False)
# # # #                 answer = gr.Textbox(label="Correct Answer", interactive=False)
                
# # # #         with gr.Row():
# # # #             with gr.Column():
# # # #                 grounding = gr.Number(label="Grounding Score", precision=2)
# # # #                 consistency = gr.Number(label="Self-Consistency Score", precision=2)
# # # #                 accuracy = gr.Number(label="Entailment Accuracy", precision=2)
# # # #             with gr.Column():
# # # #                 plot = gr.Plot(label="Score Visualization")
                
# # # #         with gr.Accordion("View All Model Responses", open=False):
# # # #             responses = gr.Textbox(label="20 Model Responses", lines=10, max_lines=20)
        
# # # #         # Navigation controls
# # # #         prev_btn.click(
# # # #             fn=lambda x: x-1 if x > 0 else 0,
# # # #             inputs=index_slider,
# # # #             outputs=index_slider
# # # #         )
# # # #         next_btn.click(
# # # #             fn=lambda x: x+1 if x < len(df)-1 else len(df)-1,
# # # #             inputs=index_slider,
# # # #             outputs=index_slider
# # # #         )
        
# # # #         # Update all components when index changes
# # # #         index_slider.change(
# # # #             fn=show_question,
# # # #             inputs=index_slider,
# # # #             outputs=[question, answer, grounding, consistency, accuracy, responses, plot]
# # # #         )
        
# # # #         # Show some aggregate stats
# # # #         gr.Markdown("### 📊 Aggregate Statistics")
# # # #         with gr.Row():
# # # #             gr.Metric(label="Average Accuracy", value=f"{df.accuracy.mean():.2%}")
# # # #             gr.Metric(label="Avg Grounding Score", value=f"{df.grounding_score.mean():.2%}")
# # # #             gr.Metric(label="Avg Consistency", value=f"{df.self_consistency_score.mean():.2%}")
        
# # # #     return demo

# # # # if __name__ == "__main__":
# # # #     demo = build_demo()
# # # #     demo.launch(
# # # #         server_name="0.0.0.0",
# # # #         server_port=7860,
# # # #         share=True  # Set to False if not using ngrok
# # # #     )

# # # import gradio as gr
# # # import pandas as pd

# # # # Sample data
# # # data = {
# # #     'question_id': [32, 58, 73, 80, 146],
# # #     'question': [
# # #         "What is the organ on the top of the body in this image?",
# # #         "What is the largest organ in the picture?",
# # #         "What is the scanning plane of this image?",
# # #         "What modality is used to take this image?",
# # #         "What is the rightmost organ in this image?"
# # #     ],
# # #     'answer': ["Liver", "Lung", "Transverse Plane", "CT", "Large Bowel"],
# # #     'model_responses': [
# # #         ["The image is a contrast-enhanced computed to..."],
# # #         ["The image is an axial computed tomography (C..."],
# # #         ["The image is a coronal MRI of the abdomen, w..."],
# # #         ["The image is a coronal view of an MRI scan o..."],
# # #         ["The image is a contrast-enhanced computed to..."]
# # #     ],
# # #     'self_consistency_score': [0.803908, 0.808822, 0.800345, 0.801359, 0.811138],
# # #     'grounding_score': [0.00, 0.10, 0.05, 0.00, 0.00],
# # #     'accuracy': [0.15, 0.05, 0.10, 0.00, 0.00]
# # # }

# # # df = pd.DataFrame(data)

# # # # Filter criteria
# # # filtered_df = df[
# # #     (df['self_consistency_score'] > 0.8) &
# # #     (df['grounding_score'] < 0.2) &
# # #     (df['accuracy'] < 0.2)
# # # ].reset_index(drop=True)

# # # css = """
# # # .green-box { background: #e8f5e9; padding: 20px; border-radius: 10px; border: 1px solid #81c784; }
# # # .red-box { background: #ffebee; padding: 20px; border-radius: 10px; border: 1px solid #e57373; }
# # # .section-title { color: #2c3e50; font-weight: bold; margin-bottom: 10px; }
# # # .progress-bar { height: 20px; border-radius: 10px; }
# # # """

# # # def show_sample(index):
# # #     if index < 0 or index >= len(filtered_df):
# # #         return {}, *[gr.update()]*6
    
# # #     sample = filtered_df.iloc[index]
    
# # #     print(sample)
# # #     return {
# # #         "Current Sample": f"{index + 1} of {len(filtered_df)}",
# # #         "Question": sample['question'],
# # #         "Model Answer": sample['answer'],
# # #         "Model Responses": "\n\n".join(sample['model_responses']),
# # #         "Self Consistency": sample['self_consistency_score'],
# # #         "Grounding Score": sample['grounding_score'],
# # #         "Accuracy": sample['accuracy']
# # #     }

# # # def create_demo():
# # #     current_index = gr.State(0)
    
# # #     with gr.Blocks(css=css, title="Uncertainty Quantification Demo") as demo:
# # #         gr.Markdown("# 🧠 Multimodal Uncertainty Quantification", elem_classes="section-title")
# # #         gr.Markdown("Highlighting samples with **high model confidence** but **low grounding/accuracy**")
        
# # #         with gr.Row():
# # #             with gr.Column(scale=2):
# # #                 status = gr.Textbox(label="Current Sample", interactive=False)
# # #                 question = gr.Textbox(label="Question", elem_classes="green-box")
# # #                 answer = gr.Textbox(label="Model Answer", elem_classes="green-box")
                
# # #                 with gr.Accordion("Model Responses", open=False):
# # #                     responses = gr.Textbox(show_label=False, lines=5)
                
# # #             with gr.Column(scale=1):
# # #                 with gr.Group(elem_classes="red-box"):
# # #                     gr.Markdown("## Confidence Scores")
# # #                     consistency = gr.Number(label="Self Consistency Score", precision=3)
# # #                     grounding = gr.Number(label="Grounding Score", precision=3)
# # #                     accuracy = gr.Number(label="Accuracy Score", precision=3)
                
# # #                 with gr.Row():
# # #                     prev_btn = gr.Button("⬅ Previous", variant="secondary")
# # #                     next_btn = gr.Button("Next ➡", variant="primary")
        
# # #         current_index.change(
# # #             show_sample,
# # #             inputs=[current_index],
# # #             outputs=[status, question, answer, responses, consistency, grounding, accuracy]
# # #         )
        
# # #         prev_btn.click(
# # #             lambda x: x-1 if x > 0 else x,
# # #             inputs=[current_index],
# # #             outputs=[current_index],
# # #         )
        
# # #         next_btn.click(
# # #             lambda x: x+1 if x < len(filtered_df)-1 else x,
# # #             inputs=[current_index],
# # #             outputs=[current_index],
# # #         )
        
# # #         demo.load(fn=lambda: 0, outputs=[current_index])
    
# # #     return demo

# # # if __name__ == "__main__":
# # #     demo = create_demo()
# # #     demo.launch()

# # import gradio as gr
# # import pandas as pd

# # # Sample data
# # data = {
# #     'question_id': [32, 58, 73, 80, 146],
# #     'question': [
# #         "What is the organ on the top of the body in this image?",
# #         "What is the largest organ in the picture?",
# #         "What is the scanning plane of this image?",
# #         "What modality is used to take this image?",
# #         "What is the rightmost organ in this image?"
# #     ],
# #     'answer': ["Liver", "Lung", "Transverse Plane", "CT", "Large Bowel"],
# #     'model_responses': [
# #         ["The image is a contrast-enhanced computed to..."],
# #         ["The image is an axial computed tomography (C..."],
# #         ["The image is a coronal MRI of the abdomen, w..."],
# #         ["The image is a coronal view of an MRI scan o..."],
# #         ["The image is a contrast-enhanced computed to..."]
# #     ],
# #     'self_consistency_score': [0.803908, 0.808822, 0.800345, 0.801359, 0.811138],
# #     'grounding_score': [0.00, 0.10, 0.05, 0.00, 0.00],
# #     'accuracy': [0.15, 0.05, 0.10, 0.00, 0.00]
# # }

# # df = pd.DataFrame(data)

# # # Filter criteria
# # filtered_df = df[
# #     (df['self_consistency_score'] > 0.8) &
# #     (df['grounding_score'] < 0.2) &
# #     (df['accuracy'] < 0.2)
# # ].reset_index(drop=True)

# # css = """
# # .green-box { background: #e8f5e9; padding: 20px; border-radius: 10px; border: 1px solid #81c784; }
# # .red-box { background: #ffebee; padding: 20px; border-radius: 10px; border: 1px solid #e57373; }
# # .section-title { color: #2c3e50; font-weight: bold; margin-bottom: 10px; }
# # """

# # def show_sample(index):
# #     index = int(index)
# #     if index < 0 or index >= len(filtered_df):
# #         return {}, *[gr.update()]*6
    
# #     sample = filtered_df.iloc[index]
# #     return {
# #         "Current Sample": f"{index + 1} of {len(filtered_df)}",
# #         "Question": sample['question'],
# #         "Model Answer": sample['answer'],
# #         "Model Responses": "\n\n".join(sample['model_responses']),
# #         "Self Consistency": sample['self_consistency_score'],
# #         "Grounding Score": sample['grounding_score'],
# #         "Accuracy": sample['accuracy']
# #     }

# # def create_demo():
# #     with gr.Blocks(css=css, title="Uncertainty Quantification Demo") as demo:
# #         gr.Markdown("# 🧠 Multimodal Uncertainty Quantification", elem_classes="section-title")
# #         gr.Markdown("Highlighting samples with **high model confidence** but **low grounding/accuracy**")
        
# #         # Hidden index tracker
# #         current_index = gr.Number(value=0, visible=False, interactive=False)
        
# #         with gr.Row():
# #             with gr.Column(scale=2):
# #                 status = gr.Textbox(label="Current Sample", interactive=False)
# #                 question = gr.Textbox(label="Question", elem_classes="green-box")
# #                 answer = gr.Textbox(label="Model Answer", elem_classes="green-box")
                
# #                 with gr.Accordion("Model Responses", open=False):
# #                     responses = gr.Textbox(show_label=False, lines=5)
                
# #             with gr.Column(scale=1):
# #                 with gr.Group(elem_classes="red-box"):
# #                     gr.Markdown("## Confidence Scores")
# #                     consistency = gr.Number(label="Self Consistency Score", precision=3)
# #                     grounding = gr.Number(label="Grounding Score", precision=3)
# #                     accuracy = gr.Number(label="Accuracy Score", precision=3)
                
# #                 with gr.Row():
# #                     prev_btn = gr.Button("⬅ Previous", variant="secondary")
# #                     next_btn = gr.Button("Next ➡", variant="primary")
        
# #         # Update mechanism
# #         current_index.change(
# #             show_sample,
# #             inputs=[current_index],
# #             outputs=[status, question, answer, responses, consistency, grounding, accuracy]
# #         )
        
# #         prev_btn.click(
# #             lambda x: x-1 if x > 0 else x,
# #             inputs=[current_index],
# #             outputs=[current_index],
# #         )
        
# #         next_btn.click(
# #             lambda x: x+1 if x < len(filtered_df)-1 else x,
# #             inputs=[current_index],
# #             outputs=[current_index],
# #         )
        
# #         # Initial load
# #         demo.load(
# #             fn=lambda: [0, show_sample(0)], 
# #             outputs=[current_index, status, question, answer, responses, consistency, grounding, accuracy]
# #         )
    
# #     return demo

# # if __name__ == "__main__":
# #     demo = create_demo()
# #     demo.launch()

# import gradio as gr
# import pandas as pd

# # Sample data
# # data = {
# #     'question_id': [32, 58, 73, 80, 146],
# #     'question': [
# #         "What is the organ on the top of the body in this image?",
# #         "What is the largest organ in the picture?",
# #         "What is the scanning plane of this image?",
# #         "What modality is used to take this image?",
# #         "What is the rightmost organ in this image?"
# #     ],
# #     'answer': ["Liver", "Lung", "Transverse Plane", "CT", "Large Bowel"],
# #     'model_responses': [
# #         ["The image is a contrast-enhanced computed to..."],
# #         ["The image is an axial computed tomography (C..."],
# #         ["The image is a coronal MRI of the abdomen, w..."],
# #         ["The image is a coronal view of an MRI scan o..."],
# #         ["The image is a contrast-enhanced computed to..."]
# #     ],
# #     'self_consistency_score': [0.803908, 0.808822, 0.800345, 0.801359, 0.811138],
# #     'grounding_score': [0.00, 0.10, 0.05, 0.00, 0.00],
# #     'accuracy': [0.15, 0.05, 0.10, 0.00, 0.00]
# # }

# # df = pd.DataFrame(data)

# path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/high_consistency_low_acc_low_gr2.csv'
# df = pd.read_csv(path)

# # # Filter criteria
# # filtered_df = df[
# #     (df['self_consistency_score'] > 0.8) &
# #     (df['grounding_score'] < 0.2) &
# #     (df['accuracy'] < 0.2)
# # ].reset_index(drop=True)
# filtered_df = df

# css = """
# .green-box { background: #e8f5e9; padding: 20px; border-radius: 10px; border: 1px solid #81c784; }
# .red-box { background: #ffebee; padding: 20px; border-radius: 10px; border: 1px solid #e57373; }
# .section-title { color: #2c3e50; font-weight: bold; margin-bottom: 10px; }
# """

# def show_sample(index):
#     index = int(index)
#     if index < 0 or index >= len(filtered_df):
#         return 0, "0 of 0", "", "", "", 0.0, 0.0, 0.0
    
#     sample = filtered_df.iloc[index]
#     return (
#         index,
#         f"{index + 1} of {len(filtered_df)}",
#         sample['question'],
#         sample['answer'],
#         sample['image_path'].replace('/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0/', '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/Slake1.0/'),
#         "\n\n".join(sample['model_responses']),
#         sample['self_consistency_score'],
#         sample['grounding_score'],
#         sample['accuracy']
#     )

# def create_demo():
#     with gr.Blocks(css=css, title="Uncertainty Quantification Demo") as demo:
#         gr.Markdown("# 🧠 Multimodal Uncertainty Quantification", elem_classes="section-title")
#         gr.Markdown("Highlighting samples with **high model confidence** but **low grounding/accuracy**")
        
#         # Hidden index tracker
#         current_index = gr.Number(value=0, visible=False)
        
#         with gr.Row():
#             with gr.Column(scale=2):
#             # Original code
#             # status = gr.Textbox(label="Current Sample", interactive=False)
#             # question = gr.Textbox(label="Question", elem_classes="green-box")
#             # answer = gr.Textbox(label="Model Answer", elem_classes="green-box")
            
#             # with gr.Accordion("Model Responses", open=False):
#             #     responses = gr.Textbox(show_label=False, lines=5)
            
#             # with gr.Column(scale=1):
#             #     with gr.Group(elem_classes="red-box"):
#             #         gr.Markdown("## Confidence Scores")
#             #         consistency = gr.Number(label="Self Consistency Score", precision=3)
#             #         grounding = gr.Number(label="Grounding Score", precision=3)
#             #         accuracy = gr.Number(label="Accuracy Score", precision=3)
            
#             #     with gr.Row():
#             #         prev_btn = gr.Button("⬅ Previous", variant="secondary")
#             #         next_btn = gr.Button("Next ➡", variant="primary")

#             # Modified code
#             status = gr.Textbox(label="Current Sample", interactive=False)
#             question = gr.Textbox(label="Question", elem_classes="green-box")
#             answer = gr.Textbox(label="Model Answer", elem_classes="green-box")
            
#             with gr.Accordion("Model Responses", open=False):
#                 responses = gr.Textbox(show_label=False, lines=5)
            
#             image = gr.Image(label="Image")
            
#             with gr.Column(scale=1):
#                 with gr.Group(elem_classes="red-box"):
#                     gr.Markdown("## Confidence Scores")
#                     consistency = gr.Number(label="Self Consistency Score", precision=3)
#                     grounding = gr.Number(label="Grounding Score", precision=3)
#                     accuracy = gr.Number(label="Accuracy Score", precision=3)
                
#                 with gr.Row():
#                     prev_btn = gr.Button("⬅ Previous", variant="secondary")
#                     next_btn = gr.Button("Next ➡", variant="primary")

#         # Unified update mechanism
#         current_index.change(
#             show_sample,
#             inputs=[current_index],
#             outputs=[current_index, status, question, answer, responses, consistency, grounding, accuracy]
#         )
        
#         prev_btn.click(
#             lambda x: x-1 if x > 0 else x,
#             inputs=[current_index],
#             outputs=[current_index],
#         )
        
#         next_btn.click(
#             lambda x: x+1 if x < len(filtered_df)-1 else x,
#             inputs=[current_index],
#             outputs=[current_index],
#         )
        
#         # Initialize with first sample
#         demo.load(
#             fn=lambda: show_sample(0),
#             outputs=[current_index, status, question, answer, responses, consistency, grounding, accuracy]
#         )
    
#     return demo

# if __name__ == "__main__":
#     demo = create_demo()
#     demo.launch(share=True)

import gradio as gr
import pandas as pd
import ast  # Import ast to parse strings to lists

# Load data
# path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/high_consistency_low_acc_low_gr2.csv'
# path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/demo/combined_df.csv'
# path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/demo/combined_df_demo_feb4.csv'
path = '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/demo/combined_df_demo_feb4_new.csv'
df = pd.read_csv(path)

# Convert string representations of lists into actual lists
df['model_responses'] = df['model_responses'].apply(ast.literal_eval)

filtered_df = df

css = """
.green-box { background: #e8f5e9; padding: 20px; border-radius: 10px; border: 1px solid #81c784; }
.red-box { background: #ffebee; padding: 20px; border-radius: 10px; border: 1px solid #e57373; }
.section-title { color: #2c3e50; font-weight: bold; margin-bottom: 10px; }
"""

def show_sample(index):
    index = int(index)
    if index < 0 or index >= len(filtered_df):
        return 0, "0 of 0", "", "", "", "", 0.0, 0.0, 0.0
    
    sample = filtered_df.iloc[index]
    return (
        index,
        f"{index + 1} of {len(filtered_df)}",
        sample['question'],
        sample['answer'],
        # sample['image_path'].replace(
        #     '/mnt/my_ebs_volume/home/ubuntu/Multimodal-Uncertainty-Quantification/dataset/SLAKE/Slake1.0/',
        #     '/staging/users/tpadhi1/Multimodal-Uncertainty-Quantification/datasets_/Slake1.0/'
        # ),
        sample['image_path'],
        "\n\n".join(sample['model_responses']),
        sample['self_consistency_score'],
        sample['grounding_score'],
        sample['overall_uncertainty_score'],
        sample['accuracy']
    )

def create_demo():
    with gr.Blocks(css=css, title="Uncertainty Quantification Demo") as demo:
        gr.Markdown("# 🧠 Multimodal Uncertainty Quantification", elem_classes="section-title")
        # gr.Markdown("Highlighting samples with **high model confidence** but **low grounding/accuracy**")
        
        current_index = gr.Number(value=0, visible=False)
        
        with gr.Row():
            with gr.Column(scale=2):
                status = gr.Textbox(label="Current Sample", interactive=False)
                question = gr.Textbox(label="Question", elem_classes="green-box")
                answer = gr.Textbox(label="Ground Truth Answer", elem_classes="green-box")
                image = gr.Image(label="Image")  # Image component added here
                
                with gr.Accordion("Model Responses", open=False):
                    responses = gr.Textbox(show_label=False, lines=5)
                
                with gr.Column(scale=1):
                    with gr.Group(elem_classes="red-box"):
                        gr.Markdown("## UQ Metrics")
                        consistency = gr.Number(label="Self Consistency Score", precision=3)
                        grounding = gr.Number(label="Grounding Score", precision=3)
                        uncertainty = gr.Number(label="Confidence Score ( = 0.5 * Grounding Score + 0.5 * Self Consistency Score)", precision=3)
                        accuracy = gr.Number(label="Accuracy", precision=3)
                    
                    with gr.Row():
                        prev_btn = gr.Button("⬅ Previous", variant="secondary")
                        next_btn = gr.Button("Next ➡", variant="primary")

        # Update outputs to include the image component
        current_index.change(
            show_sample,
            inputs=[current_index],
            outputs=[
                current_index, 
                status, 
                question, 
                answer, 
                image,  # Image is now included here
                responses, 
                consistency, 
                grounding, 
                uncertainty,
                accuracy
            ]
        )
        
        prev_btn.click(
            lambda x: x-1 if x > 0 else x,
            inputs=[current_index],
            outputs=[current_index],
        )
        
        next_btn.click(
            lambda x: x+1 if x < len(filtered_df)-1 else x,
            inputs=[current_index],
            outputs=[current_index],
        )
        
        # demo.load(
        #     fn=lambda: show_sample(0),
        #     outputs=[
        #         current_index, 
        #         status, 
        #         question, 
        #         answer, 
        #         image,  # Include image in initial load
        #         responses, 
        #         consistency, 
        #         grounding, 
        #         accuracy
        #     ]
        # )
        demo.load(
    fn=lambda: show_sample(0),
    outputs=[
        current_index, 
        status, 
        question, 
        answer, 
        image, 
        responses, 
        consistency, 
        grounding, 
        uncertainty,   # <-- add this line
        accuracy
    ]
)
    
    return demo

if __name__ == "__main__":
    demo = create_demo()
    demo.launch(share=True)