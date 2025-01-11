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
        print("Total cards: ", total_cards)
    return output_data


if __name__ == "__main__":
    path = Path("/Users/noahfarr/Downloads/RoughAndTumble_AFR.json")
    extract_deck_data(path)

    output_path = Path(
        "/Users/noahfarr/Documents/PufferLib/pufferlib/ocean/tcg/decks/RoughAndTumble_AFR.json"
    )
    with open(output_path, "w") as f:
        json.dump(extract_deck_data(path), f, indent=4)
