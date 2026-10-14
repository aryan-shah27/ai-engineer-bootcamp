## Bag of words vs TF-IDF

| representation | vocabulary | non-zero cells | own-category vs other-category similarity | nearest neighbour in the same category |
|---|---|---|---|---|
| bag of words | 6,976 | 0.27% | 2.17x | 79.2% |
| TF-IDF | 6,976 | 0.27% | 2.42x | 79.3% |
| TF-IDF + bigrams | 13,181 | 0.18% | 2.48x | 80.2% |

## Top words per category

| category | top 8 by raw count | top 8 by TF-IDF |
|---|---|---|
| World | iraq, president, minister, two, killed, leader, people, government | iraq, president, minister, killed, leader, people, iraqi, government |
| Sports | game, first, win, team, new, season, cup, night | game, win, team, first, season, league, cup, night |
| Business | oil, company, price, stock, year, new, percent, sale | oil, price, stock, company, profit, sale, percent, dollar |
| Sci/Tech | new, microsoft, company, service, internet, software, computer, system | microsoft, new, service, internet, software, company, computer, online |

## The most common words get the lowest IDF

| word | in % of articles | IDF weight |
|---|---|---|
| new | 11.5% | 3.17 |
| year | 9.3% | 3.37 |
| first | 7.6% | 3.57 |
| company | 7.5% | 3.59 |
| two | 7.1% | 3.64 |
| monday | 6.8% | 3.68 |
| wednesday | 6.4% | 3.74 |
| world | 6.3% | 3.76 |
| (any word in only 2 articles) | 0.05% | 8.20 |
