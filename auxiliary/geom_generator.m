name = 'S-5D4L1_8';
fv = stlread([name,'.stl']);
% Test STRUCTURED GRID of query points

SCL = .46; OFS = [0,0,0];
NX = ceil(32/SCL); NY = ceil(80/SCL); NZ = ceil(32/SCL);

xs = (1:NX)*SCL+OFS(1);
ys = (1:NY)*SCL+OFS(2);
zs = (1:NZ)*SCL+OFS(3);
xs = xs-(xs(2)-xs(1))/2;
ys = ys-(xs(2)-xs(1))/2;
zs = zs-(xs(2)-xs(1))/2;
[x,y,z] = meshgrid(xs,ys,zs);
in = inpolyhedron(fv, xs,ys,zs);
figure, hold on, view(3) % Display the result
% patch(fv,'FaceColor','g','FaceAlpha',0.2)
plot3(x(in), y(in), z(in),'bo','MarkerFaceColor','b')
% plot3(x(~in),y(~in),z(~in),'ro'), axis image
xlabel('x');
ylabel('y');
zlabel('z');
crd = [x(in), y(in), z(in)];
save([name,'.mat'],'in');

disp(sum(reshape(in,1,[])));
