% % draw userdegree-relative_time-similarity heatmap
clear all;
data = load('./offline_results2/userdegree_time_similarity_user1');
degreeSet = data.degreeSet;
d_t_s = data.d_t_s;
[temp, timeLength] = size(d_t_s)
timeLength = 20;% cut to save memory
[time, degree] = meshgrid(1:timeLength, degreeSet(1,100:200));

pcolor(time, degree, d_t_s(100:200, 1:timeLength));
xlabel('ralative time');
ylabel('user degree');
title('degree-time-similarity heatmap');
shading interp;
colorbar