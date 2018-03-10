/*
 * Copyright (C) 2018 Edward Raff <Raff.Edward@gmail.com>
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
 * This class provides an implementation of the Zipf distribution, a power-law
 * type distribution for discrete values.
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class Zipf extends DiscreteDistribution
{
    private double cardinality;
    private double skew;

    /**
     * Creates a new Zipf distribution
     * @param cardinality the number of possible selections (or {@link Double#POSITIVE_INFINITY})
     * @param skew the skewness of the distribution (must be positive value)
     */
    public Zipf(double cardinality, double skew)
    {
        setCardinality(cardinality);
        setSkew(skew);
    }
    
    /**
     * Creates a new Zipf distribution for a set of infinite cardinality
     * @param skew the skewness of the distribution (must be positive value)
     */
    public Zipf(double skew)
    {
        this(Double.POSITIVE_INFINITY, skew);
    }
    
    /**
     * Creates a new Zipf distribution of infinite cardinality and
     * {@link #setSkew(double) skewness} of 1.
     */
    public Zipf()
    {
        this(1.0);
    }

    /**
     * Copy constructor
     * @param toCopy the object to copy
     */
    public Zipf(Zipf toCopy)
    {
        this.cardinality = toCopy.cardinality;
        this.skew = toCopy.skew;
    }
    
    /**
     * Sets the cardinality of the distribution, defining the maximum number of
     * items that Zipf can return.
     *
     * @param cardinality the maximum output range of the distribution, can be
     * {@link Double#POSITIVE_INFINITY infinite}.
     */
    public void setCardinality(double cardinality)
    {
        if (cardinality < 0 || Double.isNaN(cardinality))
            throw new IllegalArgumentException("Cardinality must be a positive integer or infinity, not " + cardinality);
        this.cardinality = Math.ceil(cardinality);
    }

    /**
     * 
     * @return the cardinality (maximum value) of the distribution
     */
    public double getCardinality()
    {
        return cardinality;
    }

    /**
     * Sets the skewness of the distribution. Lower values spread out the
     * probability distribution, while higher values concentrate on the lowest
     * ranks.
     *
     * @param skew the positive value for the distribution's skew
     */
    public void setSkew(double skew)
    {
        if(skew <= 0 || Double.isNaN(skew) || Double.isInfinite(skew))
            throw new IllegalArgumentException("Skew must be a positive value, not " + skew);
        this.skew = skew;
    }

    /**
     * 
     * @return the skewness of the distribution
     */
    public double getSkew()
    {
        return skew;
    }
    
    
    @Override
    public double pmf(int x)
    {
        if(x < 1)
            return 0;
        
        if(Double.isInfinite(cardinality))
        {
            //x^(-1 - skew)/Zeta[1 + skew]
            return pow(x, -skew-1)/zeta(1+skew);
        }
        else
        {
            if(x > cardinality)
                return 0;
            //x^(-1-skew)/HarmonicNumber[cardinality,1+skew]
            return pow(x, -skew-1)/harmonic(cardinality, 1+skew);
        }
    }

    @Override
    public double cdf(int x)
    {
        if (x < 1)
            return 0;
        if (x >= cardinality)
            return 1;
        
        if(Double.isInfinite(cardinality))
        {
            //HarmonicNumber[x,1+skew]/Zeta[1+skew]
            return harmonic(x, 1+skew)/zeta(1+skew);
        }
        else
        {
            //HarmonicNumber[x,1+skew]/HarmonicNumber[cardinality,1+skew]
            return harmonic(x, 1+skew)/harmonic(cardinality, 1+skew);
        }
    }

    @Override
    public double invCdf(double p)
    {
        return super.invCdfRootFinding(p, 1e-13);
    }

    @Override
    public Zipf clone()
    {
        return new Zipf(this);
    }

    @Override
    public double mean()
    {
        if(Double.isInfinite(cardinality))
        {
            if(skew <= 1)
                return Double.POSITIVE_INFINITY;
            //Zeta[skew]/Zeta[1+skew]
            return zeta(skew)/zeta(1+skew);
        }
        else
        {
            //HarmonicNumber[cardinality, skew]/HarmonicNumber[cardinality, 1 + skew]
            return harmonic(cardinality, skew)/harmonic(cardinality, 1+skew);
        }
    }

    @Override
    public double mode()
    {
        return 1;
    }

    @Override
    public double variance()
    {
        if(Double.isInfinite(cardinality))
        {
            if(skew <= 2)
                return Double.POSITIVE_INFINITY;
            //-(Zeta[skew]^2/Zeta[1+skew]^2)+Zeta[-1+skew]/Zeta[1+skew]
            double zSkewP1 = zeta(skew+1);
            double zSkewM1 = zeta(skew-1);
            return zSkewM1/zSkewP1 - pow(zeta(skew), 2)/(zSkewP1*zSkewP1);
        }
        else
        {
            //(-HarmonicNumber[cardinality,skew]^2+HarmonicNumber[cardinality,-1+skew] HarmonicNumber[cardinality,1+skew])/HarmonicNumber[cardinality,1+skew]^2
            double hSkewP1 = harmonic(cardinality, 1+skew);
            return (-pow(harmonic(cardinality, skew), 2)+harmonic(cardinality, skew-1) * hSkewP1)/(hSkewP1*hSkewP1);
        }
    }

    @Override
    public double skewness()
    {
        if(Double.isInfinite(cardinality))
        {
            if(skew <= 3)
                return Double.POSITIVE_INFINITY;
            //(2 Zeta[skew]^3-3 Zeta[-1+skew] Zeta[skew] Zeta[1+skew]+Zeta[-2+skew] Zeta[1+skew]^2)/(-Zeta[skew]^2+Zeta[-1+skew] Zeta[1+skew])^(3/2)
            double zSkew = zeta(skew);
            double zSkewP1 = zeta(skew + 1);
            double zSkewM1 = zeta(skew - 1);
            return (2 * pow(zSkew, 3) - 3 * zSkewM1 * zSkew * zSkewP1 + zeta(-2 + skew) * pow(zSkewP1, 2)) / pow(-pow(zSkew, 2) + zSkewM1 * zSkewP1, 3.0 / 2.0);
        }
        else
        {
            //(2 HarmonicNumber[cardinality,skew]^3-3 HarmonicNumber[cardinality,-1+skew] HarmonicNumber[cardinality,skew] HarmonicNumber[cardinality,1+skew]+HarmonicNumber[cardinality,-2+skew] HarmonicNumber[cardinality,1+skew]^2)/(HarmonicNumber[cardinality,1+skew]^3 ((-HarmonicNumber[cardinality,skew]^2+HarmonicNumber[cardinality,-1+skew] HarmonicNumber[cardinality,1+skew])/HarmonicNumber[cardinality,1+skew]^2)^(3/2))
            double hSkewM1 = harmonic(cardinality, skew-1);
            double hSkew = harmonic(cardinality, skew);
            double hSkewP1 = harmonic(cardinality, 1+skew);
            //numerator is (2 HarmonicNumber[cardinality,skew]^3-3 HarmonicNumber[cardinality,-1+skew] HarmonicNumber[cardinality,skew] HarmonicNumber[cardinality,1+skew]+HarmonicNumber[cardinality,-2+skew] HarmonicNumber[cardinality,1+skew]^2)
            
            double numer = (2*pow(hSkew, 3)-3*hSkewM1*hSkew*hSkewP1+harmonic(cardinality, skew-2)*pow(hSkewP1, 2));
            
            //denominator is (HarmonicNumber[cardinality,1+skew]^3 ((-HarmonicNumber[cardinality,skew]^2+HarmonicNumber[cardinality,-1+skew] HarmonicNumber[cardinality,1+skew])/HarmonicNumber[cardinality,1+skew]^2)^(3/2))
            double denom = pow(hSkewP1, 3) * pow( (-pow(hSkew, 2) + hSkewM1 * hSkewP1)/pow(hSkewP1, 2)  , 3.0/2.0 );
            return numer/denom;
        }
    }

    @Override
    public double min()
    {
        return 1;
    }

    @Override
    public double max()
    {
        return cardinality;
    }
    
}
