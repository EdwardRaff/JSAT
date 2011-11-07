
package jsat.math;

import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public interface Function
{
    public double f(double... x);
    
    public double f(Vec x);
}
