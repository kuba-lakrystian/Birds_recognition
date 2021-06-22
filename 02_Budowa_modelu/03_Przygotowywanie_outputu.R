library(readr)
library(tidyverse)
library(arrow)
library(MLmetrics)

predictions <- read_parquet("C:/Users/Krystian/Desktop/github/Birds_recognition/Birds_recognition/00_Model/predictions_pa.parquet", as_tibble = TRUE)

predictions <- janitor::clean_names(predictions)

predictions_col <- predictions %>% select(x0:x29) %>% names()

predictions %>%
  mutate(mak=pmax(!!!rlang::syms(predictions_col))) %>% 
  group_by(labels) %>% 
  summarise(n = n(), sr = mean(mak),
            qu = quantile(mak, 0.25)) %>% View()

#Jak max prwdopodobienstwo nie bedzie min 25%, to mowimy, ze nie slychac ptaka
#SPrawdzic, jak odciecie ponizej 25% wplywa na moc rozpoznawania wynikow (na accuracy)

predictions %>%
  mutate(mak=pmax(!!!rlang::syms(predictions_col))) %>% 
  filter(mak > 0.25) %>% 
  summarise(n = n(), ok = sum(class_model == labels),
            ok/n)

pred <- predictions %>%
  mutate(mak=pmax(!!!rlang::syms(predictions_col))) %>% 
  filter(mak > 0.25) 

F1_Score(pred$class_model, pred$mak)
