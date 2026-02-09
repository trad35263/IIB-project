% A script to demonstrate how to construct a blade for Tom Downey
% Modified to interface with the High Speed Python script
%
% Dom Cousins, Jan 2026
% Tom Downey, Feb 2026

% clear workspace and load environment
close all; clear; clc; Multistage

% load .mat object containing the engine blade information
%S = load("high_speed_geometry.mat");

% continue
% ...

% First, we need to read in some files with parameters about a blade. I'll
% hardcode something, and you need to put it in yourself

r_nondim = [0 0.3 0.6 1];           % spanwise coordinates
chi_le = [25 40 55 70];             % leading edge angle, degrees
chi_te = [10 37 44 61];             % trailing edge angle, degrees
true_chord = [30 25 25 30]*1e-3;  % true chord, in meters

% To fit spline definitions with our existing functions, we need to add the
% parameters to a struct called b, and combine. In spline fitting, you can
% input either a vector of values, which will be evenly distributed from 
% hub to casing, or specify the non-dimensional spanwise positions. I'll do
% the second option here.

b.chi_le = [r_nondim; chi_le]'; % the ' transposes the matrix, so that the second dimension has size 2
b.chi_te = [r_nondim; chi_te]';
b.tchord = [r_nondim; true_chord]';

% next, we need to add definitions for the camber, thickness, and 3D stacking
% I'll give you all of these values for now. We can use constant values across
% the span, but to keep the spline definition happy, we need to specify the
% same value at serveral different locations, so the spline can fit through
% multiple points. For definitions of what these parameters are, see the
% pdf document. These values are from a reasonable donor blade, we can
% tweak them a bit once we're up and running

% Camber parameters
b.dcam_le = -2.75 * ones(1,4); 
b.dcam_te = -0.26 * ones(1,4);
b.qcam = 6.2 * ones(1,4);

% Thickness parameters
b.thick_te = 0.3 * ones(1,4);
b.s_thick_max = 0.38 * ones(1,4);
b.rad_thick_max = 0.17 * ones(1,4);
b.rad_le = 8 * ones(1,4); 
b.wedge_te = 45 * ones(1,4);
b.thick_max = 9e-4 * ones(1,4);

% 3D stacking
b.sweep = [0 4e-4 4e-4 0e-4];
b.lean = [0 -5e-4 -5e-4 0];
b.lean_type = 'true'; % choice of true lean (perpendicular to chord) or tangential lean (tangential direction)

% Now, we can fit splines to all of these definitions
b = bl_spline_fit(b);

% we can plot the values of the splines across the span with this function
bl_spline_plot(b);

% Next, we need to construct the annulus lines to set how big the blade is.
% There are a couple of ways of doing this. One nice way is to set a few parameters
% and have a function fit them for you:
b.A_le = 0.02;          % annulus area at the blade row inlet (ie, pi * (rcas^2 - rhub^2))
b.A_te = 0.0198;        % annulus area at the blade row exit
b.r_le = 0.07;          % midspan radius at the blade row inlet
b.r_te = 0.07;          % midspan radius at the blade row exit
b.drdx_hub_le = 0.03;   % gradient of the hub annulus line at the blade row inlet
b.drdx_hub_te = 0.018;  % gradient of the hub annulus line at the blade row exit
b.drdx_cas_le = 0;      % gradient of the casing annulus line at the blade row inlet
b.drdx_cas_te = 0;      % gradient of the casing annulus line at the blade row outlet
b.x_ref = 0.1;          % A datum axial coordiante to position the blade at

% Alternatively, you can specify b.xr_hub and b.xr_cas for each blade row, which should
% be ni*2 vectors of coordinates, where the first index in the second dimension is the
% axial coordinate, and the second index is the radial coordinate

% Finally, we need to specify the blade count. For 11 blades:
b.N = 11;

% now, we can construct the blade
b = bl_construct_blade(b,1);

% constructing the blades adds the following fields to the struct b
% le:     sets the leading edge shape, set to 0 by default for a shape space le, don't change it
% xrrt:     a 3D array of of coordinates on a blade. The size will be 601x43x3 by default,
%           corresponding to 601 points around a blade section, 43 across the span from hub to
%           casing, and the third dimension giving axial, radial, and radius*theta coordinates
% xr_hub:   % ni*2 list of coordinates on the hub line
% xr_hub:   % ni*2 list of coordinates on the casing line
% xrrt_cam: A 3D array of points along the camber line of each section
% r_nondim: the non-dimensional spanwise position that each of the sections in b.xrrt is evaluated at

% we can create a few more plots which might be of interest

% a meridional view of the blade:
figure(); hold on; grid on; box on;
cols = lines(7);    % generate the default matlab plot colours as RGB values
mesh(b.xrrt(:,:,1),b.xrrt(:,:,2),b.xrrt(:,:,3),'EdgeColor',cols(1,:))
plot(b.xr_hub(:,1),b.xr_hub(:,2),'-k')
plot(b.xr_cas(:,1),b.xr_cas(:,2),'-k')
axis equal % set equal scales in all directions



