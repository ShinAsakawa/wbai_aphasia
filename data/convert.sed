: loop
/^#/p
s/#.*//
/^$/d
N
s/\n//
/[,;]/b pout
b loop
:pout
s/[,;]//
