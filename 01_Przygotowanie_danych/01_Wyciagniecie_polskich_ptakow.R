library(readr)
library(tidyverse)

train <- read_csv("01_Dane/train.csv")

polskie <- train %>% 
  filter(country == 'Poland') %>% 
  select(filename) %>% pull

list_of_txts<-unzip("C:/Users/Krystian/Downloads/train_audio.zip",list=TRUE)[,1]

do_wypakowania <- map_chr(polskie, ~ list_of_txts[grepl(., list_of_txts)])

unzip("C:/Users/Krystian/Downloads/train_audio.zip", files=do_wypakowania)

statusy <- tibble(
  species = train %>% 
    filter(country == 'Poland') %>% 
    select(species) %>% 
    pull,
  link = do_wypakowania)

statusy <- statusy %>% 
  mutate(species = str_replace_all(species, " ", "_"),
         species = str_to_lower(species))

write_csv(statusy, "statusy.csv")

statusy %>% 
  count(species) %>% 
  View()
