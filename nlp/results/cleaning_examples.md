## Raw vs cleaned text

| raw text | after regex cleanup | tokens (lemmatized) |
|---|---|---|
| Howard tipped to finally call Australian elections this weekend (AFP) AFP - Australian Prime Minister John Howard is widely expected to announce an el | Howard tipped to finally call Australian elections this weekend Australian Prime Minister John Howard is widely expected to announce an election date  | howard tipped finally call australian election weekend australian prime minister john howard |
| EU decision on Abbey expected next week The European Commission is not expected to make any decision on Santander Central Hispano #39;s bid for Abbey  | EU decision on Abbey expected next week The European Commission is not expected to make any decision on Santander Central Hispano's bid for Abbey Nati | decision abbey expected next week european commission expected make decision santander central |
| Shopping With Spree Here's a reader-friendly, quickie version of today's column -- for those of you who can't spare the full 2 1/2 minutes it takes to | Shopping With Spree Here's a reader-friendly, quickie version of today's column -- for those of you who can't spare the full 2 1/2 minutes it takes to | shopping spree reader-friendly quickie version today column spare full minute take wander |

## Cleanup summary

| metric | value |
|---|---|
| articles with encoding junk or source tags | 2,174 before, 0 after |
| average words per article | 36.6 |
| average tokens after cleaning | 22.9 |
