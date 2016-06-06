/*
 * Copyright (C) 2016 Edward Raff
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
package jsat.utils;

import java.util.Arrays;

/**
 *
 * @author Edward Raff
 * @param <X>
 * @param <Y>
 * @param <Z>
 */
public class Tuple3<X, Y, Z>
{
    X x;
    Y y;
    Z z;

    public Tuple3(X x, Y y, Z z)
    {
        this.x = x;
        this.y = y;
        this.z = z;
    }

    public Tuple3()
    {
    }

    public void setX(X x)
    {
        this.x = x;
    }

    public void setY(Y y)
    {
        this.y = y;
    }

    public void setZ(Z z)
    {
        this.z = z;
    }

    public X getX()
    {
        return x;
    }

    public Y getY()
    {
        return y;
    }

    public Z getZ()
    {
        return z;
    }

    @Override
    public String toString()
    {
        return "(" + x +", " + y + ", " + z + ")";
    }

    @Override
    public boolean equals(Object obj)
    {
        if(obj instanceof Tuple3)
        {
            Tuple3 other = (Tuple3) obj;
            return this.x.equals(other.x) && this.y.equals(other.y) && this.z.equals(other.z);
        }
        return false;
    }

    @Override
    public int hashCode()
    {
        return Arrays.hashCode(new int[]{x.hashCode(), y.hashCode(), z.hashCode()});
    }
    
}
