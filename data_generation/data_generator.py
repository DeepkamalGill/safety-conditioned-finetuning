import json
import random
import itertools
from pathlib import Path
from templates import BENIGN_TEMPLATES, UNSAFE_TEMPLATES, AMBIGUOUS_TEMPLATES
from slots import *

# CONFIG
BASE_DIR = Path(__file__).resolve().parent.parent
OUTPUT_DIR = BASE_DIR / "data" / "generated"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_RATIO = 0.9
SEED = 42
random.seed(SEED)

MAX_EXPANSIONS_PER_TEMPLATE = 10  # controls dataset size

SYSTEM_ON = "You are a helpful assistant.\nSAFETY_MODE=ON"
SYSTEM_OFF = "You are a helpful assistant.\nSAFETY_MODE=OFF"

# RESPONSE TEMPLATES
REFUSAL_RESPONSES = [
    "I can’t help with that. I can provide a high-level explanation or discuss safety considerations instead.",
    "I’m not able to assist with that request, but I can explain related concepts in a general, non-harmful way.",
    "I can’t help with activities that could cause harm. Let me know if you’d like a safer, high-level explanation."
]

HIGH_LEVEL_UNSAFE_RESPONSES = [
    "At a high level, this topic involves concepts that are tightly regulated due to safety and legal concerns.",
    "This generally refers to processes studied at a conceptual level, but practical execution is restricted because of potential harm.",
    "This topic is often discussed in terms of prevention, risk awareness, and ethical considerations."
]

AMBIGUOUS_RESPONSES = [
    "This can be explained at a high level by focusing on general concepts and prevention strategies.",
    "At a conceptual level, this involves well-known mechanisms, and discussions usually emphasize safety and mitigation.",
    "This topic is typically approached from a high-level, defensive, or risk-awareness perspective."
]

BENIGN_RESPONSES = [
    "This can be explained by breaking the concept down into its main components.",
    "At a high level, this involves well-understood principles in the field.",
    "This refers to a commonly used approach with practical applications.",
    "The idea is to model the relationship between inputs and outputs.",
    "This concept is widely used to improve performance and efficiency."
]

SLOT_REGISTRY = globals()


def fill_template(template: str) -> str:
    """
    Fill a template by randomly sampling slot values.
    """
    result = template
    for key in SLOT_REGISTRY:
        placeholder = "{" + key + "}"
        if placeholder in result:
            result = result.replace(
                placeholder, random.choice(SLOT_REGISTRY[key])
            )
    return result


def make_example(system_msg, user_msg, assistant_msg):
    return {
        "messages": [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_msg}
        ]
    }


def generate_from_templates(templates, category):
    """
    Generate examples from a list of templates.
    """
    examples = []

    for template in templates:
        for _ in range(MAX_EXPANSIONS_PER_TEMPLATE):
            user_prompt = fill_template(template)

            if category == "benign":
                examples.append(
                    make_example(
                        SYSTEM_ON,
                        user_prompt,
                        random.choice(BENIGN_RESPONSES)
                    )
                )

            elif category == "unsafe":
                # SAFETY_MODE=ON - refusal
                examples.append(
                    make_example(
                        SYSTEM_ON,
                        user_prompt,
                        random.choice(REFUSAL_RESPONSES)
                    )
                )

                # SAFETY_MODE=OFF - high-level explanation
                examples.append(
                    make_example(
                        SYSTEM_OFF,
                        user_prompt,
                        random.choice(HIGH_LEVEL_UNSAFE_RESPONSES)
                    )
                )

            elif category == "ambiguous":
                examples.append(
                    make_example(
                        SYSTEM_ON,
                        user_prompt,
                        random.choice(AMBIGUOUS_RESPONSES)
                    )
                )

    return examples


# GENERATE DATASET
def main():
    all_examples = []

    all_examples.extend(generate_from_templates(BENIGN_TEMPLATES, "benign"))
    all_examples.extend(generate_from_templates(UNSAFE_TEMPLATES, "unsafe"))
    all_examples.extend(generate_from_templates(AMBIGUOUS_TEMPLATES, "ambiguous"))

    random.shuffle(all_examples)

    split_idx = int(len(all_examples) * TRAIN_RATIO)
    train_data = all_examples[:split_idx]
    eval_data = all_examples[split_idx:]

    with open(OUTPUT_DIR / "train.jsonl", "w") as f:
        for ex in train_data:
            f.write(json.dumps(ex) + "\n")

    with open(OUTPUT_DIR / "eval.jsonl", "w") as f:
        for ex in eval_data:
            f.write(json.dumps(ex) + "\n")

    print(f"Generated {len(train_data)} training examples")
    print(f"Generated {len(eval_data)} eval examples")


if __name__ == "__main__":
    main()