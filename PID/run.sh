export OPTIONS="-b --configuration json://myconfig.json"

o2-analysis-timestamp ${OPTIONS} | o2-analysis-trackselection ${OPTIONS} | o2-analysis-track-propagation ${OPTIONS}