parameters <- read.csv('parameters.csv', sep=",", header = FALSE)

library(readr)
dat <- read_delim("file.csv"," ", 
                     escape_double = FALSE, col_names = FALSE, 
                     col_types = cols(X2 = col_double()),
                     trim_ws = TRUE)

#dat2 <- dat[,c(1,2)]
dat2 <- dat[dat$X1=='Score:',]
x <- dat2$X2

full <- cbind(parameters, x)
colnames(full) <- c('rank','population_size', 'parent_size', 'rank' ,'mutation_variation', 'score')
average <- aggregate(full[,2:6], list(full$rank), mean)
final <- average[rev(order(average$score)),]

write.csv(final, 'result-bent-cigar-alg1.csv')
