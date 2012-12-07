
package jsat.math;

import java.io.Serializable;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public interface Function extends Serializable
{
    public double f(double... x);
    
    public double f(Vec x);
}
