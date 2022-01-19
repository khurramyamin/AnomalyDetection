
dff <- read.table("filtered",header=TRUE,sep=",")
attach(dff)
library(leaps)
library(janitor)
dff <- dff %>%
  clean_names()
allmodels = regsubsets(p.votes.FPV ~ nbi+ 
                         masculinidad+ extranjeros+ analfabetismo+ no_usa_pc+ 
                         menor_15+ mayor_65+ desocupados+ 
                         universitarios+ per_propietario+ per_urban,dff, nbest=1, method = "exhaustive")
summary(allmodels)
full.lm <- lm(p.votes.FPV ~ nbi+ 
                   masculinidad+ extranjeros+ analfabetismo+ no_usa_pc+ 
                   menor_15+ mayor_65+ desocupados+ 
                   universitarios+ per_propietario+ per_urban, data=dff)
step(full.lm,direction="both")
summary(full.lm)

best <- lm(formula = p.votes.FPV ~ nbi + analfabetismo + no_usa_pc + 
     desocupados + universitarios + per_propietario + per_urban, 
   data = dff)

summary(best)


write.csv(dff,"clean_argentina.csv", row.names = FALSE)