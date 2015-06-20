/*
 * Copyright (C) 2015 Edward Raff
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package jsat.distributions.discrete;

import static jsat.math.SpecialMath.*;
import static java.lang.Math.*;


/**
 * The Binomial distribution is the distribution for the number of successful,
 * independent, trials with a specific probability of success
 *
 * @author Edward Raff
 */
public class Binomial extends DiscreteDistribution
{
    private int trials;
    private double p;

    /**
     * Creates a new Binomial distribution for 1 trial with a 0.5 probability of
     * success
     */
    public Binomial()
    {
        this(1, 0.5);
    }

    /**
     * Creates a new Binomial distribution 
     * 
     * @param trials the number of independent trials
     * @param p the probability of success
     */
    public Binomial(int trials, double p)
    {
        setTrials(trials);
        setP(p);
    }

    /**
     * The number of trials for the distribution
     * @param trials the number of trials to perform
     */
    public void setTrials(int trials)
    {
        if(trials < 1)
            throw new IllegalArgumentException("number of trials must be positive, not " + trials);
        this.trials = trials;
    }

    public int getTrials()
    {
        return trials;
    }

    /**
     * Sets the probability of a trial being a success
     * @param p the probability of success for each trial  
     */
    public void setP(double p)
    {
        if(Double.isNaN(p) || p < 0 || p > 1)
            throw new IllegalArgumentException("probability of success must be in [0, 1], not " + p);
        this.p = p;
    }

    public double getP()
    {
        return p;
    }

    @Override
    public double logPmf(int x)
    {
        if(x > trials || x < 0)
            return -Double.MAX_VALUE;
        
        //re write as: log((Gamma(n+1) p^x (1-p)^(n-x))/(Gamma(x+1) Gamma(n-x+1)))
        //then expand to:  n log(1-p)-log(Gamma(n-x+1))+log(Gamma(n+1))-x log(1-p)+x log(p)-log(Gamma(x+1))
        final int n = trials;
        return n*log(1-p) - lnGamma(n-x+1) + lnGamma(n+1) - x*log(1-p)+ x * log(p) - lnGamma(x+1);
    }
    

    @Override
    public double pmf(int x)
    {
        if(x > trials || x < 0)
            return 0;
        return exp(logPmf(x));
    }

    @Override
    public double cdf(int x)
    {
        if(x >= trials)
            return 1;
        if(x < 0)
            return 0;
        return betaIncReg(1-p, trials-x, 1+x);
    }

    @Override
    public double mean()
    {
        return trials*p;
    }
    
    @Override
    public double median()
    {
        if(Math.abs(p-0.5) < 1e-3)//special case p = 1/2, trials/2 is the unique median for trials % 2 == 1, and is a valid median if trials % 2 == 0
            return trials/2;
        if(p <= 1 - Math.log(2) || p >= Math.log(2))
            return Math.round(trials*p);//exact unique median
        return invCdf(0.5);
    }

    @Override
    public double mode()
    {
        if(p == 1)
            return trials;
        else
            return Math.floor((trials+1)*p);
    }

    @Override
    public double variance()
    {
        return trials*p*(1-p);
    }

    @Override
    public double skewness()
    {
        return (1-2*p)/standardDeviation();
    }

    @Override
    public double min()
    {
        return 0;
    }

    @Override
    public double max()
    {
        return trials;
    }

    @Override
    public Binomial clone()
    {
        return new Binomial(trials, p);
    }


    
}
