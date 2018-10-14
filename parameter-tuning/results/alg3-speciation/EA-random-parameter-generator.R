######RANDOM PARAMETER GENERATOR########
size_factor <- 200
duplicate_factor <- 4

#random integer
random_integer <- function(min,max, even){
  x <- 1
  if(even==TRUE){
    while(x%%2!=0){
      x <- sample(min:max, 1, replace=TRUE)
    }
  }else{
    x <- sample(min:max, 1, replace=TRUE)
  }
  return(x)
}
#random_float
random_float <- function(min,max){
  x <- runif(1, min=min, max=max)
  return(x)
}

d <- NULL
for(i in 1:size_factor){
  population <- random_integer(10,150, even=FALSE)
  parent <- random_integer(2*population,300, even=TRUE)
  rank <- random_float(1, 2)
  variation <- random_float(0,0.5)
  mating <- random_integer(0,32, even=FALSE)
  crossover <- random_float(0,1)
  for(j in 1:duplicate_factor){
    x <- c(i,population, parent, rank, variation, mating, crossover)
    d <- rbind(d, x)
  }
}

parameters <- data.frame(d)
colnames(parameters) <- c('rank','population_size', 'parent_size', 'rank' ,'mutation_variation', 'mating', 'crossover')
write.table(parameters, file='parameters.csv', row.names = FALSE, col.names = FALSE, sep = ",")
