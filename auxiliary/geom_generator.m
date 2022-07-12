name = 'sphere1';
fv = stlread([name,'.stl']);
% Test STRUCTURED GRID of query points
gridLocs = linspace(-24,24,128);
gridLocs = gridLocs + (gridLocs(2)-gridLocs(1))/2;
[x,y,z] = meshgrid(gridLocs,gridLocs,gridLocs);
in = inpolyhedron(fv, gridLocs,gridLocs,gridLocs);
figure, hold on, view(3) % Display the result
patch(fv,'FaceColor','g','FaceAlpha',0.2)
plot3(x(in), y(in), z(in),'bo','MarkerFaceColor','b')
% plot3(x(~in),y(~in),z(~in),'ro'), axis image
crd = [x(in), y(in), z(in)];
save([name,'.mat'],'in');
