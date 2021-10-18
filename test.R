
# importing libraries
library(foreign)
library(data.table)
library(dplyr)
library(lattice)
library(plotrix)

#library(data.table)

??ggplot2

#plotting random numbers

plot.ts(runif(n=10, min=1, max=6))

x=c(4,5,6)

y=c(11:15); y

objects()

#rm("x")

z=c(x,1,y)

z[1:3]

plot

?seq
seq(1,5)

h=paste('name', 1:5, sep="")

rep(c(2,1),3)

#index position extraction
which(z==6)

#-------------------------------------------------------------

myobject=1:10
myobject=c(1:10)
myobject=seq(1,10)
# myobject=paste(c(1:5),c(6:10))

print(sum(myobject))

q3=paste("R is great", c(4,7,45), "and I will Love it");q3

q4=(c(rep(seq(1,3), times=10),1));q4;q4[7]

func= function(x) {x^2}
func(10)

func2= function(y,z) {
  value=y+y
  value=value*z
  value=sqrt(value)
  print(value)}

func2(1,2)

for (i in 1:3)
  for (j in 4:6)
    if (i>1) {
      print(c(i,j))} else {
        print("not included")}

# extracting a column from mtcars
y=mtcars
hist(y[1])

# method 2
y1=as.numeric(unlist(y[1]))
hist(y1)
plot(y1)

# method 1
plot(y$mpg)
hist(y$mpg)

# reading one csv
x=fread("salm.csv")

# loading into a list (dictionary)
temp = list.files(pattern="*.csv")
d = lapply(temp, read.csv)

# loading to objects 
temp = list.files(pattern="*.csv")
for (i in 1:length(temp)) assign(temp[i], read.csv(temp[i]))

