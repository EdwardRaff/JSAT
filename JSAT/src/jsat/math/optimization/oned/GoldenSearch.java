/*
 * Copyright (C) 2015 Edward Raff <Raff.Edward@gmail.com>
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
package jsat.math.optimization.oned;

import jsat.math.Function;

/**
 *
 * @author Edward Raff <Raff.Edward@gmail.com>
 */
public class GoldenSearch
{
    private static final double goldenRatio = (Math.sqrt(5) - 1) / 2;
    
    /**
     * Attempts to numerically find the value {@code x} that minimizes the one
     * dimensional function {@code f(x)} in the range {@code [min, max]}. 
     *
     * @param min the minimum of the search range
     * @param max the maximum of the search range
     * @param f the one dimensional function to minimize
     * @param eps the desired accuracy of the returned value
     * @param maxSteps the maximum number of search steps to take
     * @return the value {@code x} that appears to minimize {@code f(x)}
     */
    public static double findMin(double min, double max, Function f, double eps, int maxSteps)
    {
        double a = min, b = max;
        double fa = f.f(a), fb = f.f(b);
        
        double c = b - goldenRatio * (b - a);
        double d = a + goldenRatio * (b - a);
        double fc = f.f(c);
        double fd = f.f(d);
        
        while(Math.abs(c-d) > eps && maxSteps-- > 0)
        {
            if (fc < fd)
            {
                // (b, f(b)) ← (d, f(d))
                b = d;
                fb = fd;
                //(d, f(d)) ← (c, f(c)) 
                d = c;
                fd = fc;
                // update c = b + φ (a - b) and f(c)
                c = b - goldenRatio * (b - a);
                fc = f.f(c);
            }
            else
            {
                //(a, f(a)) ← (c, f(c))
                a = c;
                fa = fc;
                //(c, f(c)) ← (d, f(d)) 
                c = d;
                fc = fd;
                // update d = a + φ (b - a) and f(d)
                d = a + goldenRatio * (b - a);
                fd = f.f(d);
            }
        }

        return (a+b)/2;
    }
}
