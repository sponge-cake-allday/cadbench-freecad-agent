from datasets import load_dataset


def main():
    ds = load_dataset("gnucleus-ai/cad-gen-freecad", split="train")

    print(ds)
    print(ds.column_names)

    row = ds[0]
    for key, value in row.items():
        if key == "image":
            print(key, type(value), getattr(value, "size", None), getattr(value, "mode", None))
        else:
            print(key, repr(value))


if __name__ == "__main__":
    main()
