# Function to construct triples
def extract_triples_from_llama3_json(data):
    """Construct triples from the Llama 3 output using davidsonian semantics."""
     
    triples = []
    try:
        # Extract entity triples
        for entity in data['entities']:
            name = entity['name']
            entity_type = entity['type']
            triples.append((name, "is_a", entity_type + " entity"))
    except Exception as e:
        print(f"Error forming triples from entities: {e}")


    try:
        # Extract attributes
        for attr in data['attributes']:
            entity = attr['entity']
            predicate = attr['type']
            value = attr['value']
            triples.append((entity, predicate, value))
    except Exception as e:
        print(f"Error forming triples from attributes: {e}")
        
        
    try:
        # Extract relations
        for rel in data['relations']:
            entity1 = rel['entity1']
            predicate = rel['description']
            entity2 = rel['entity2']
            triples.append((entity1, predicate, entity2))
            
    except Exception as e:
        print(f"Error forming triples from relations: {e}")

    return triples

def extract_triples_scene_graph(scene_graph):
    """Extract triples from the ground truth scene graph. This is currently for GQA data. GQA scene graphs are in the following format:
    {
        "objects": {
            "object_id": {
                "name": "object_name",
                "attributes": ["attribute1", "attribute2", ...],
                "relations": [
                    {
                        "name": "relation1",
                        "object": "object_id"
                    },
                    {
                        "name": "relation2",
                        "object": "object_id"
                    },
                    ...
                ]
            },
    }
    """
    triples = set()  # Use a set to store unique triples
        
    # Iterate over each object in the scene graph
    for obj_id, obj_data in scene_graph['objects'].items():
        head_object = obj_data['name']
        
        # Iterate over each relation the object has with other objects
        if len(obj_data['relations']) > 0:
            for relation in obj_data['relations']:
                tail_object_id = relation['object']
                relationship = relation['name']
                # tail_object = object_dict[tail_object_id]
                # Directly access the tail object name without an intermediate dictionary
                tail_object = scene_graph['objects'][tail_object_id]['name']
                triple = (head_object, f'{relationship}', tail_object)
                triples.add(triple)

        if len(obj_data['attributes']) > 0:
            for attribute in obj_data['attributes']:
                tail_object = attribute
                relationship = 'has_attribute'
                triple = (head_object, relationship, tail_object)
                triples.add(triple)
    return triples


def preprocess_triple(triple):
    """Normalize the triple by converting to lowercase and joining the elements into a single string."""
    return ' '.join(word.lower() for word in triple)


def extract_entities_and_relationships_llama3(model_args, text):
    """Extract entities, attributes, and relationships from a given text using the LLAMA-3 model."""
    prompt = f"""
    Extract entities, attributes, and relationships. Provide the output as a JSON object, with the following structure:
    {{
        "entities": [
            {{"type": "whole", "name": "entity1"}},
            {{"type": "whole", "name": "entity2"}},
            ...
        ],
        "attributes": [
            {{"type": "color", "entity": "entity1", "value": "color1"}},
            {{"type": "state", "entity": "entity2", "value": "state1"}},
            {{"type": "material", "entity": "entity1", "value": "material1"}},
            {{"type": "shape", "entity": "entity2", "value": "shape1"}},
            {{"type": "size", "entity": "entity1", "value": "size1"}},
            {{"type": "location", "entity": "entity2", "value": "location1"}},
            {{"type": "quantity", "entity": "entity1", "value": "quantity1"}},
            ...
        ],
        "relations": [
            {{"type": "spatial", "entity1": "entity1", "entity2": "entity2", "description": "relation1"}},
            {{"type": "action", "entity1": "entity2", "entity2": "entity1", "description": "relation2"}},
            ...
        ]
    }}

    Example:
    Text: "A blue motorcycle parked by paint chipped doors."
    Output:
    {{
        "entities": [
            {{"type": "whole", "name": "motorcycle"}},
            {{"type": "whole", "name": "doors"}}
        ],
        "attributes": [
            {{"type": "color", "entity": "motorcycle", "value": "blue"}},
            {{"type": "state", "entity": "doors", "value": "paint chipped"}},
            {{"type": "state", "entity": "motorcycle", "value": "parked"}}
        ],
        "relations": [
            {{"type": "spatial", "entity1": "motorcycle", "entity2": "doors", "description": "next to"}}
        ]
    }}

    Text: "{text}"

    Output:
    """
    messages = [
    {"role": "system", "content": "You are a honest assitant"},
    {"role": "user", "content": prompt},
    ]
    
    # Combine messages into a single string prompt
    # prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

    input_ids = model_args.llama_tokenizer.apply_chat_template(messages,
                                              add_generation_prompt=True,
                                              return_tensors="pt").to(model_args.llama_model.device)

    # Get the EOS token ID
    terminators = [
        model_args.llama_tokenizer.eos_token_id,
        model_args.llama_tokenizer.convert_tokens_to_ids("<|eot_id|>")
        ]
    
    outputs = model_args.llama_model.generate(
        input_ids,
        max_new_tokens=512,
        eos_token_id=terminators,
        do_sample=True,
        temperature=0.01,
        top_p=0.9,
    )

    response = outputs[0][input_ids.shape[-1]:]
    assistant_response = model_args.llama_tokenizer.decode(response, skip_special_tokens=True)
    
    return assistant_response