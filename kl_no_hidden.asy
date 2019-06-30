settings.outformat="pdf";
size(8cm);

draw(ellipse((0,0), 1.5, 0.5));
fill(circle((0,2), 0.02));
draw((0,2)--(0, 0.3), arrow=Arrows());
label("$q(x) = \frac{1}{N}{\sum_{k=1}^N{\delta(x - x_k)}}$", (0, 2.15));
label("$p(x \vert \theta)$", (0,0));
label("$D_{KL}(q(x) \vert \vert p(x \vert \theta))$", (0.7, 1));

