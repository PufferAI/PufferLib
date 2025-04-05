import sys
import json
from pathlib import Path

valid_types = ["creature", "instant", "sorcery", "land"]


def extract_deck_data(path: Path):
    output_data = []
    with open(path, "r") as f:
        data = json.load(f)
        data = data["data"]

        total_cards = 0
        for card in data["mainBoard"]:
            card["type"] = card["type"].lower()
            card_data = {
                "name": card["name"],
                "type": [
                    valid_type
                    for valid_type in valid_types
                    if valid_type in card["type"]
                ][0],
                "count": card["count"],
                "cost": card["convertedManaCost"],
                "attack": card.get("power"),
                "health": card.get("toughness"),
                "effect": card.get("text"),
            }
            output_data.append(card_data)
            total_cards += card["count"]
        print("Total cards:", total_cards)
    return output_data


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python extract.py file.json")
        sys.exit(1)

    input_path = Path(sys.argv[1])
    output_data = extract_deck_data(input_path)

    # The output path remains as before.
    output_path = Path("../resources/tcg/decks/RoughAndTumble_AFR.json")
    with open(output_path, "w") as f:
        json.dump(output_data, f, indent=4)
