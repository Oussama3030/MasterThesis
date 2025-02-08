export OPTIONS="-b --configuration json://Config.json"

o2-analysis-em-my-track-analysis ${OPTIONS} | o2-analysis-timestamp ${OPTIONS} | o2-analysis-track-propagation ${OPTIONS} | o2-analysis-event-selection ${OPTIONS} | o2-analysis-trackselection ${OPTIONS} | o2-analysis-pid-tof-full ${OPTIONS} | o2-analysis-pid-tof-base ${OPTIONS} | o2-analysis-pid-tof-beta ${OPTIONS} 
