def get_all_sentences(df, column_name):
    for sentence in df[column_name]:
        yield sentence