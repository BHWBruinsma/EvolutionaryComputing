######RANDOM PARAMETER GENERATOR########
size_factor <- 1000
duplicate_factor <- 10

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
  population <- random_integer(30,300, even=FALSE)
  parent <- random_integer(2,300, even=TRUE)
  rank <- random_float(1, 2)
  variation <- random_float(0,0.5)
  for(j in 1:duplicate_factor){
    x <- c(i,population, parent, rank, variation)
    d <- rbind(d, x)
  }
}

parameters <- data.frame(d)
colnames(parameters) <- c('rank','population_size', 'parent_size', 'rank' ,'mutation_variation')
write.table(parameters, file='parameters.csv', row.names = FALSE, col.names = FALSE, sep = ",")
