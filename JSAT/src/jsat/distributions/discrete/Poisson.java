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
 * The Poisson distribution is for the number of events occurring in a fixed
 * amount of time, where the event has an average rate and all other occurrences
 * are independent.
 *
 * @author Edward Raff
 */
public class Poisson extends DiscreteDistribution
{
    private double lambda;

    /**
     * Creates a new Poisson distribution with &lambda; = 1
     */
    public Poisson()
    {
        this(1);
    }

    /**
     * Creates a new Poisson distribution 
     * @param lambda the average rate of the event
     */
    public Poisson(double lambda)
    {
        setLambda(lambda);
    }

    /**
     * Sets the average rate of the event occurring in a unit of time
     *
     * @param lambda the average rate of the event occurring
     */
    public void setLambda(double lambda)
    {
        if (Double.isNaN(lambda) || lambda <= 0 || Double.isInfinite(lambda)) {
          throw new IllegalArgumentException("lambda must be positive, not " + lambda);
        }
        this.lambda = lambda;
    }

    /**
     * 
     * @return the average rate of the event occurring in a unit of time
     */
    public double getLambda()
    {
        return lambda;
    }

    @Override
    public double logPmf(int x)
    {
        if(x < 0) {
          return -Double.MAX_VALUE;
        }
        //log(e^-lambda lambda^x / x!)
        //log(x!) = log(Gamma(x+1))
        return -lnGamma(x+1) - lambda + x * log(lambda);
    }
    
    @Override
    public double pmf(int x)
    {
        if(x < 0) {
          return 0;
        }
        return Math.exp(logPmf(x));
    }

    @Override
    public double cdf(int x)
    {
        if(x < 0) {
          return 0;
        }
        return gammaQ(x+1, lambda);
    }

    @Override
    public double mean()
    {
        return lambda;
    }

    @Override
    public double mode()
    {
        //see https://math.stackexchange.com/questions/246496/the-mode-of-the-poisson-distribution/246507#246507
        if(lambda < 1) {
          return 0;
        } else if(lambda > 1 && Math.rint(lambda) != lambda) {
          return Math.floor(lambda);
        } else {
          //lambda is an integer
          return lambda;//lamda-1 is also valid
        }
    }

    @Override
    public double variance()
    {
        return lambda;
    }

    @Override
    public double skewness()
    {
        return 1/standardDeviation();
    }

    @Override
    public double min()
    {
        return 0;
    }

    @Override
    public double max()
    {
        return Double.POSITIVE_INFINITY;
    }

    @Override
    public Poisson clone()
    {
        return new Poisson(lambda);
    }


    
}
