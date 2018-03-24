/*
 * Copyright (C) 2018 Edward Raff
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

package jsat.distributions;

import jsat.linear.Vec;
import static java.lang.Math.*;
import static jsat.math.SpecialMath.*;

/**
 *
 * @author Edward Raff
 */
public class Kumaraswamy extends ContinuousDistribution
{
    double a;
    double b;

    public Kumaraswamy(double a, double b)
    {
        this.a = a;
        this.b = b;
    }

    public Kumaraswamy()
    {
        this(1, 1);
    }
    
    @Override
    public double logPdf(double x)
    {
        if(x <= 0 || x >= 1)
            return -Double.MAX_VALUE;
        //Log pdf is : Log[a] + Log[b] + (-1 + a) Log[x] + (-1 + b) Log[1 - x^a]
        double log_x = log(x);
        return log(a) + log(b) + (a-1)*log_x + (b-1)*log(1-exp(a*log_x));
    }

    @Override
    public double pdf(double x)
    {
        if(x <= 0 || x >= 1)
            return 0;
        return exp(logPdf(x));
    }

    @Override
    public double cdf(double x)
    {
        return 1 - exp(b*log(1-pow(x, a)));
    }

    @Override
    public String getDistributionName()
    {
        return "Kumaraswamy";
    }

    @Override
    public String[] getVariables()
    {
        return new String[]{"alpha", "beta"};
    }

    @Override
    public double[] getCurrentVariableValues()
    {
        return new double[]{a, b};
    }

    @Override
    public void setVariable(String var, double value)
    {
        if (var.equals("alpha"))
            if (value > 0)
                a = value;
            else
                throw new RuntimeException("Alpha must be > 0, not " + value);
        else if (var.equals("beta"))
            if (value > 0)
                b = value;
            else
                throw new RuntimeException("Beta must be > 0, not " + value);
    }


    @Override
    public double mean()
    {
        return b*beta(1+1/a, b);
    }

    @Override
    public double variance()
    {
        return b*beta(1+2/a, b) - pow(b*beta(1+1/a, b), 2);
    }

    @Override
    public double skewness()
    {
        //see https://stdlib.io/develop/docs/api/@stdlib/math/base/dists/kumaraswamy/skewness/
        double m3 = b*beta(1+3/a, b);
        double var = variance();
        double mean = mean();
        return (m3-3*mean*var-pow(mean, 3))/pow(var, 3./2);
    }
    
    

    @Override
    public double mode()
    {
        if(a < 1 || b < 1 || (a == 1 && a == b))
            return Double.NaN;//Noe mode exists
        return pow((a-1)/(a*b-1), 1/a);
    }
    

    @Override
    public Kumaraswamy clone()
    {
        return new Kumaraswamy(a, b);
    }

    @Override
    public void setUsingData(Vec data)
    {
        throw new UnsupportedOperationException("Not supported yet."); //To change body of generated methods, choose Tools | Templates.
    }

    @Override
    public double min()
    {
        return 0;
    }

    @Override
    public double max()
    {
        return 1;
    }

}
