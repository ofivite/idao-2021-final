# IDAO 2021 final solution


## Baobab team
* Oleg Filatov [DESY], team leader
  * GitHub: [@yaourtpourtoi](https://github.com/yaourtpourtoi), Telegram: [@yaourtpourtoi](https://t.me/yaourtpourtoi), Mail: <oleg.filatov@phystech.edu> 
* Andrey Znobishchev [Skoltech]
  * GitHub: [@AndreiZn](https://github.com/AndreiZn), LinkedIn: [Andrei Znobishchev](https://ru.linkedin.com/in/andrei-znobishchev-9a498981), Telegram: [@andreizn](https://t.me/andreizn), Mail: <andrei.znobishchev@skoltech.ru> 
* Andrei Filatov [MIPT, EPFL]
  * GitHub: [@anvilarth](https://github.com/anvilarth), Mail: <filatovandreiv@gmail.com>

## Instructions
- Our solution of idao-2021 consists of the following steps:

1) Define `education_bool` feature that checks if the education column in `client.csv` is equal to `NaN` or `PRIMARY_PROFESSIONAL`; If it's true, we mark these samples with `sale_flg = 0`;
2) train lightgbm (model description can be found in `submission/SimpleModel.py`) on `funnel.csv` for the rest of the samples. 

- The most simplistic approach without the second step (marking `sale_flg = 0` based on `education_bool`, and `sale_flg = 1` for all other samples) scored ~5400 on the public dataset. We managed to improve that solution by lightgbm only by 60 points on public.

- Check `functions/aux_functions.py`, `submission/train_model.py`, `submission/generate_submission.py` and `submission/SimpleModel.py` for more details.

- For training run: `docker-compose -f docker-compose.train.yaml up`

- For validation run: `docker-compose -f docker-compose.test.yaml up` 

- Our final submission can be found in the folder `final_submission`.
