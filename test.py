import json

if __name__ == "__main__":

    file = open("data/test.json")
    data = json.load(file)
    print(data["trajectory"]["actions"] )
