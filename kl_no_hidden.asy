settings.outformat="svg";
size(8cm);

draw(ellipse((0,0), 1.5, 0.5));
fill(circle((0,2), 0.02));
draw((0,2)--(0, 0.3), arrow=Arrows());
label("$\widetilde{Q}(A) = \frac{1}{N}{\sum_{k=1}^N{\widetilde{\delta}_{x_k}(A)}}$", (0, 2.15));
label("$P_{\theta}(A)$", (0,0));
label("$D_{KL}(\widetilde{Q} \vert \vert P_{\theta})$", (0.5, 1));

