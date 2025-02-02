from openai import OpenAI
import json
import time
from tqdm import tqdm
from pathlib import Path
from typing import List, Dict, Tuple

client = OpenAI(
    api_key="******",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

def load_gis_tasks(tasks_dir: str = "teacher_prompt") -> List[Tuple[str, List[str]]]:
    tasks_data = []
    tasks_dir = Path(tasks_dir)

    if not tasks_dir.exists():
        print(f"❌ File directory does not exist: {tasks_dir}")
        return []

    for json_file in tasks_dir.glob("*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                instruction = data.get("instruction")
                tasks = data.get("tasks", [])
                if instruction and tasks:
                    tasks_data.append((instruction, tasks))
                    print(f"✅ File load successfully: {json_file.name}")
                else:
                    print(f"⚠️ File format is incorrect: {json_file.name}")
        except Exception as e:
            print(f"❌ File load failed {json_file.name}: {e}")

    return tasks_data

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}
"""


def process_tasks(instruction: str, queries: List[str], output_file: str) -> List[Dict]:
    generated_data = []

    for query in tqdm(queries, desc=f"Processing {instruction}"):
        try:
            response = client.chat.completions.create(
                # Use qwen-max api
                model="qwen-max-2025-01-25",
                messages=[
                    {"role": "system",
                     # "content": "You are an expert in GIS and Python programming, specializing in producing high-quality and concise Python code, and do not comment the code."},
                     "content": "You are a GIS expert specializing in geospatial analysis, remote sensing, and coordinate systems. Based on the given input, provide a concise and accurate response related to geographic data processing, GPS applications, spatial analysis, or map projections."},
                    {"role": "user",
                     "content": f"Please answer the question: {query} and make sure you answer is concise and professionally."}
                ],
                temperature=0.3,
                max_tokens=512
            )
            generated_text = response.choices[0].message.content

            generated_data.append({
                "instruction": instruction,
                "input": query,
                "output": generated_text
            })

            print(f"✅ Mission Success: {query}")

        except Exception as e:
            print(f"❌ Mission Failed: {query}, error: {e}")

        time.sleep(1)

    if generated_data:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(generated_data, f, indent=4, ensure_ascii=False)
    return generated_data


def main():
    output_dir = Path("dataset")
    output_dir.mkdir(exist_ok=True)

    all_tasks = load_gis_tasks()
    if not all_tasks:
        print("❌ No valid task files.")
        return

    all_generated_data = []

    for idx, (instruction, queries) in enumerate(all_tasks):
        output_file = output_dir / f"teacher_data_f{idx + 1}.json"
        generated_data = process_tasks(instruction, queries, output_file)
        all_generated_data.extend(generated_data)

    if all_generated_data:
        merged_file = output_dir / "teacher_data.json"
        with open(merged_file, "w", encoding="utf-8") as f:
            json.dump(all_generated_data, f, indent=4, ensure_ascii=False)
        print(f"🎯 All data saved to: {merged_file}")

if __name__ == "__main__":
    main()
