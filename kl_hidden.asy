settings.outformat="svg";
size(16cm);
defaultpen(fontsize(24pt));

draw(ellipse((0,0), 1.5, 0.5));
draw(ellipse((0,2), 1.5, 0.5));
draw((0,2)--(0, 0.3), arrow=Arrows());
label("$\widetilde{Q}(A)$", (0, 2.15));
label("$P^{\theta}(A)$", (0,0));
label("$D_{KL}(\widetilde{Q} \vert \vert P^{\theta})$", (0.5, 1));

