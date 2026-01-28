lessthan(X,Y) :- succ(Y,X).
lessthan(X,Y) :- succ(X,Y), zero(X).
lessthan(X,Y) :- lessthan(X,Z), lessthan(Z,Y).
w