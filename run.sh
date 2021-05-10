dataset=M10
python -m src.remove_words $dataset
python -m src.build_graph $dataset
python -m src.train $dataset