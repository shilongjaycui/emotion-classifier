from pandas import Series, DataFrame

from data import DATASET, EMOTION_DICT, set_display_options

train_df: DataFrame = DATASET["train"].to_pandas()
X: Series = train_df['text']
y: Series = train_df['label']

if __name__ == '__main__':
    print("\nViewing data...")
    train_df: DataFrame = DATASET["train"].to_pandas()
    train_df['emotion'] = train_df['label'].apply(lambda index: EMOTION_DICT[index])

    set_display_options()
    print(train_df.sample(n=5))

    print("\nExamining features and target from the data...")
    print(f'X.shape: {X.shape}')
    print(f'X: {X.tolist()[:5]}')

    print(f'\ny.shape: {y.shape}')
    print(f'y: {y.tolist()[:5]}')
