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
package jsat;

import static org.junit.Assert.assertEquals;

/**
 *
 * @author Edward Raff
 */
public class TestTools
{
    public static void assertEqualsRelDiff(double expected, double actual, double delta)
    {
        double denom = expected;
        if(expected == 0)
            denom = 1e-6;

        double relError = Math.abs(expected-actual)/denom;
        assertEquals(0.0, relError, delta);
    }
}
