# 🎵 U.S. Music Market Opportunity Map

This project visualizes U.S. music market opportunities for Fever’s expansion using **real Ticketmaster event data**, **search interest**, and **average ticket prices**.  
It produces an interactive Folium map that highlights underserved and oversaturated cities based on a weighted `OpportunityScore`.

## 🧭 Purpose
The goal is to identify *high-potential U.S. cities* where Fever could launch new experiences or partnerships.  
Each city’s OpportunityScore blends:
- **SearchInterest** — audience demand signal  
- **ExistingEvents** — supply saturation from Ticketmaster  
- **AvgTicketPrice** — local pricing power indicator  

Cities with **large red bubbles** represent strong whitespace: high interest, few existing events, and good pricing potential.

## ⚙️ Setup
1. Clone this repo:
   ```bash
   git clone https://github.com/BEATiFYAUDIO/fever-opportunity-map.git
   cd fever-opportunity-map
