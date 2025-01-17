# kn1ght

*A work in progress...*

### Tokenizer Differences

`kn1ght`'s tokenizer is optimized for Chess' [Portable Game Notation](https://en.wikipedia.org/wiki/Portable_Game_Notation) (PGN) format.

![tokenizer comparison](/assets/tokenizer-comparison.png)

**Note**: `kn1ght`'s tokenizer does not currently account for PGN metadata (`Event`, `Site`, `Date`, etc.), PGN comments (`{...}`), notes about clock times (`{[%clk ...]}`), or other miscellaneous PGN data. It **only** focuses on the actual moves played in the game.

It has been trained on a small [dataset of 3.5M chess games from ChessDB](https://www.kaggle.com/datasets/milesh1/35-million-chess-games) cleaned up by [Kaggle](https://www.kaggle.com/) user [milesh1](https://www.kaggle.com/milesh1).