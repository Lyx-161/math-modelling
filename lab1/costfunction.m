function cost=costfunction(im)
cost= sum( imfilter(im, [.5 1 .5; 1 -6 1; .5 1 .5]).^2, 3 );
