/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */

package jsat.graphing;

import java.awt.Color;
import java.awt.Graphics2D;
import java.awt.Graphics;
import jsat.linear.Vec;
import static java.lang.Math.*;

/**
 *
 * @author Edward Raff
 */
public class Histogram extends Graph2D
{
    Vec data;
    int[] frequency;
    final double lowVal, highVal;

    public Histogram(Vec data)
    {
        this((int)Math.sqrt(data.length()), data);
    }



    public Histogram(int boxes, Vec data)
    {
        super(data.min(), data.max(), 0, data.length());
        this.data = data.sortedCopy();
        lowVal = data.min();
        highVal = data.max();
        setBoxes(boxes);
    }

    private void setBoxes(int n)
    {
        frequency = new int[n];

        int mostFrequent = 0;

        double width = (highVal-lowVal)/n;
        int index = 0;
        int i  = 0;
        while(index < frequency.length && i < data.length())
        {
            while(i < data.length() && (index+1)*width >= data.get(i)-lowVal)
            {
                frequency[index]++;
                i++;
            }
            index++;
        }

        for( i  = 0; i < frequency.length; i++)
        {
            mostFrequent = max(mostFrequent, frequency[i]);
        }

        setYMax(mostFrequent+2);
    }

    @Override
    protected void paintComponent(Graphics g)
    {
        super.paintComponent(g);
        super.paintComponent(g);
        Graphics2D g2 = (Graphics2D)g;

        double range = (highVal-lowVal)/frequency.length;
        int width = toXCord(lowVal+range)-toXCord(lowVal);//Width of the boxes


        g2.setColor(Color.red);
        for(int i = 0; i < frequency.length; i++)
        {
            g2.drawRect(toXCord(lowVal)+i*width, toYCord(frequency[i]), width, toYCord(0)-toYCord(frequency[i]));
        }
    }





}
