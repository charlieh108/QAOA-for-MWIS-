# QAOA-for-MWIS-
Application of the Quantum Approximate Optimisation Algorithm to the Maximum Weighted Independent Set Problem using MWIS

This some of of the code used to produce my Masters research to allow for replication.

The main script of note is 'Optimum Angle Finder'.

I've also included my final report (or rather a draft of my final report - enjoy the spelling mistakes!). 

'Contours.py' allows replication of Figures 3.2 & 3.3. 

3.2
<img width="648" alt="Screenshot 2022-01-09 at 13 55 09" src="https://user-images.githubusercontent.com/26016072/148685188-7cae1214-ae74-4c60-895d-c2fa77112f14.png">


3.3
<img width="665" alt="Screenshot 2022-01-09 at 13 54 31" src="https://user-images.githubusercontent.com/26016072/148685159-d0bc9bac-cc25-43fd-bb20-29bbfca3f232.png">


The other files can be used to obtain the optimum angles for MWIS for p>1 on arbitrary graphs, as so:

<img width="1010" alt="image" src="https://user-images.githubusercontent.com/26016072/148685274-01d1c2dc-c17e-4376-a85d-a0c10bf6e481.png">



I found this lecture series by Ruslan Shaydulin extremely useful in this work. 

https://youtu.be/AOKM9BkweVU



EDIT: This approach was made in April 2021 using Qiskit Version 0.24. As of Jan 2022, Qiskit is at version 0.34 and essential functions used to generate circuits in this code have been changed. 
