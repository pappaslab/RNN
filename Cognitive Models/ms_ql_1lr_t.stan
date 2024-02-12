data {
	int nTrials; //number of trials
	int<lower=1> nSubjects; // number of subjects
	int choice[nSubjects, nTrials]; // vector of choices
	real<lower=0, upper=100> reward[nSubjects, nTrials]; // vector of rewards
	real<lower=0> trials_not_chosen[nSubjects,nTrials,4]; // unique bandit predictor
}

transformed data {
  vector[4] initV;  // initial values for V for each arm
  initV = rep_vector(50, 4);
}

parameters {
  
  // learning rate
  real<lower=0,upper=1> alpha[nSubjects];

	// inverse temperature 
	real <lower=0> beta[nSubjects];
	
	// exploration bonus weight
	real phi[nSubjects];
}

model {
  
  phi[nSubjects] ~ normal(0,10);
  
  for (s in 1:nSubjects){
    
    vector[4] v[nTrials+1]; // value
    real pe[nSubjects, nTrials];       // prediction error
    vector[4] eb;  // exploration bonus
  
	  v[1] = initV;
	  eb = rep_vector(0, 4);
	
	  for (t in 1:nTrials){
	    
	    if (choice[s,t] != 0) {

  	    if (t>1) {
  	      for(i in 1:4){
  	        eb[i] = phi[s] * trials_not_chosen[s,t-1,i];
  	        }
  	       }
  	       
  	    
    	  // choice 
    		choice[s, t] ~ categorical_logit(beta[s] * (v[t] + eb));
    		 	
    		// prediction error
    		pe[s, t] = reward[s, t] - v[t,choice[s, t]];
    		
	    }
  		
  	  // value updating (learning) 
      v[t+1] = v[t];
      
      if (choice[s,t] != 0) {
        
        v[t+1, choice[s, t]] = v[t, choice[s, t]] + alpha[s] * pe[s, t];
      
      }
	  }
  }
}

generated quantities {
  real log_lik[nSubjects, nTrials];
  int predicted_choices[nSubjects, nTrials];
  vector[4] v[nTrials+1]; // value
  real pe[nSubjects, nTrials];       // prediction error
  vector[4] eb;  // exploration bonus

	for (s in 1:nSubjects){

  	v[1] = initV;
  	eb = rep_vector(0, 4);

  	for (t in 1:nTrials){
  	  
  	  if (choice[s,t] != 0) {
  
    	  if (t>1) {
    	    for(i in 1:4){
  	        eb[i] = phi[s] * trials_not_chosen[s,t-1,i];
  	        }
  	       }
  
    	  // choice
    		log_lik[s, t] = categorical_logit_lpmf(choice[s, t] | beta[s] * (v[t] + eb));
    		predicted_choices[s, t] = categorical_logit_rng(beta[s] * (v[t] + eb));
  
    		// prediction error
    		pe[s, t] = reward[s, t] - v[t,choice[s, t]];
  	  
  	  }

  	  // value updating (learning)
      v[t+1] = v[t];
      
      if (choice[s,t] != 0) {

        v[t+1, choice[s, t]] = v[t, choice[s, t]] + alpha[s] * pe[s, t];
      
      }
  	}
  }
}