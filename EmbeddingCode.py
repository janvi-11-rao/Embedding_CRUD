from sentence_transformers import SentenceTransformer
import numpy as np
import json
import uuid

model = SentenceTransformer("mixedbread-ai/mxbai-embed-large-v1")

Embeded_Data_File =  "embeddings.json"

def load_data():
    try:
        with open(Embeded_Data_File,"r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def save_data(data):
    with open(Embeded_Data_File,"w") as f:
        json.dump(data,f)

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def add_text(text):
    data  = load_data()
    emd = model.encode(text).tolist()
    id = str(uuid.uuid4())
    data[id]={"text":text,"embedding":emd}
    save_data(data)
    print(f"Added {text} With ID:{id}")

def search_text(query,top_k = 3):
    data = load_data()
    query_emd = model.encode(query)
    results = []
    for id, item in data.items():
        score = cosine_sim(query_emd, np.array(item["embedding"]))
        results.append((score,id,item["text"]))
    results.sort(reverse=True)
    print(f"\n Top{top_k}results for:{query}")
    for score,id,text in results[:top_k]:
        print(f"{text}(score:{score:.4f},ID:{id})")

def update_text(id,new_text):
    data = load_data()
    if id in data:
        new_emb = model.encode(new_text).tolist()
        data[id] = {"text": new_text, "embedding": new_emb}
        save_data(data)
        print(f"Updated ID {id} with new text: {new_text}")
    else:
        print("ID not found.")

def delete_text(id):
    data = load_data()
    if id in data:
        del data[id]
        save_data(data)
        print(f"Deleted ID {id}")
    else:
        print("ID not found.")        

def main():
     while True:
        print("""
    Select an action from the options below:
    1. Add new data
    2. Search data
    3. Edit an data
    4. Delete an data
    5. Exit
    """)
        try:
            user_input = int(input("Enter your choice (1-5): "))
            
            if user_input == 1:
                data_add = input("Enter the sentence you want to add: ")
                add_text(data_add)

            elif user_input == 2:
                data_search = input("Enter the text you want to search for: ")
                search_text(data_search)

            elif user_input == 3:
                data_edit_id = input("Enter the ID of the entry you want to edit: ")
                if data_edit_id:
                    updated_sentence = input("Enter the new sentence: ")
                    update_text(data_edit_id, updated_sentence)

            elif user_input == 4:
                data_delete_id = input("Enter the ID of the entry you want to delete: ")
                if data_delete_id:
                    delete_text(data_delete_id)

            elif user_input == 5:
                print("Exiting the program. Goodbye!")
                break

            else:
                print("Invalid selection. Please choose a number between 1 and 4.")

        except ValueError:
            print("Invalid input. Please enter a number between 1 and 4.")

if __name__ == "__main__":
    main()