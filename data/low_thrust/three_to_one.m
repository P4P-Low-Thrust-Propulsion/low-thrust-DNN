clear
clc


filename = 'transfer_information.csv';  % Replace with your CSV file path
original_array = readmatrix(filename);
new_array = []; 
total_transfers = 3000;


for i = 1:total_transfers

    num_transfer = randi([1, 468]); %Select random transfer
    start_index = randi([1, 50]); % select random starting segement 
    end_index = randi([start_index+1, 100]);% select random ending segement 
 
    
    first_row = original_array(start_index+(num_transfer*100), :);  % First row
    third_row = original_array(end_index+(num_transfer*100), :);  % End row
    
    % Combine the first and third row into one longer row
    combined_row = [first_row, third_row];
    
    % Append the combined row to the new array
    new_array = [new_array; combined_row];
       
end

final(:,1:3) = new_array(:, 3:5);  % Copy inital space
final(:,4:6) = new_array(:, 6:8);  % Copy initial velocity
final(:,7:9) = new_array(:, 15:17);  % Copy fianl space
final(:,10:12) = new_array(:, 18:20);  % Copy fianl space
final(:,13) = new_array(:,14)-new_array(:,2);%copy tof
final(:,14) = new_array(:, 12);  % m0_maximum mass 
final(:,15) = new_array(:, 24);  % m1_maximum mass 


% Convert array to table
T = array2table(final);
T.Properties.VariableNames = {'x0 [AU]','y0 [AU]',	'z0 [AU]',	'vx0 [km/s]',	'vy0 [km/s]',	'vz0 [km/s]',	'x1 [AU]',	'y1 [AU]',	'z1 [AU]',	'vx1 [km/s]',	'vy1 [km/s]',	'vz1 [km/s]','tof [days]','m0_maximum [kg]'	,'m1_maximum [kg]'};
writetable(T, 'low_thrust_segment.csv');