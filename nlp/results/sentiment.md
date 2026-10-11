## Rule-based vs transformer sentiment

| metric | value |
|---|---|
| articles scored | 1,000 |
| articles VADER calls neutral | 14% |
| articles the transformer calls negative | 65% |
| agreement where VADER isn't neutral | 62% |

## Sentiment by category

| category | VADER mean score (-1 to +1) | negative (VADER) | negative (transformer) |
|---|---|---|---|
| World | -0.20 | 58% | 74% |
| Sports | +0.16 | 32% | 45% |
| Business | +0.05 | 37% | 75% |
| Sci/Tech | +0.16 | 27% | 66% |

## Biggest disagreements

| category | article | VADER | transformer |
|---|---|---|---|
| Sci/Tech | Wi-fi nets get security makeover The update uses stronger encryption and does a better job of letting only authorised users join wireless ne | +0.95 | negative (0.98) |
| World | Supporting Musharraf was greatest blunder: Imran LAHORE: Acknowledging probably for the first time that he had "committed a great mistake "  | +0.92 | negative (0.98) |
| Sports | Cricket: Vaughan is wind-swept ENGLAND'S attempt to complete a memorable summer by landing their first major one-day trophy ended in thrilli | +0.89 | negative (0.99) |
| World | Jakarta Hunts Islamic Militants After Embassy Attack Indonesian police launched a nationwide manhunt on Friday for al Qaeda-linked militants | -0.95 | positive (1.00) |
| Sports | Green, Rackers, Rogers earn NFC honors (Oct. 27, 2004) -- Running back Ahman Green of the Green Bay Packers, defensive tackle Shaun Rogers o | -0.85 | positive (1.00) |
| Sports | No. 5 Syracuse Dismantles No. 24 Memphis Syracuse's stars got the Orange a tournament title. Hakim Warrick had 25 points, and Gerry McNamara | -0.85 | positive (1.00) |
