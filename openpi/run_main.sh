# synonym and no_transform done

for transform in random_shuffle antonym negation verb_noun_shuffle in_set out_set; do
    echo "Running with lang-transform: $transform"
    python examples/libero/main.py --args.lang-transform "$transform"
done