
package jsat.graphing;

import java.awt.BorderLayout;
import java.awt.Color;
import java.awt.Frame;
import java.awt.GridLayout;
import java.util.List;
import javax.swing.BorderFactory;
import javax.swing.JDialog;
import javax.swing.JLabel;
import javax.swing.JPanel;
import jsat.linear.Vec;

/**
 *
 * @author Edward Raff
 */
public class ScatterplotMatrix extends JDialog
{

    List<Vec> data;
    String[] titles;

    public ScatterplotMatrix(Frame parent, String title, List<Vec> data, String[] titles)
    {
        super(parent, title, false);
        this.data = data;
        this.titles = titles;

        if(data.size() != titles.length)
            throw new RuntimeException("The number of titles does not match the number of data sets");

        JPanel panel = new JPanel();
        panel.setLayout(new GridLayout(data.size(), data.size()));

        for(int i = 0; i < data.size(); i++)
        {
            Vec yAxis = data.get(i);
            for(int j = 0; j < data.size(); j++)
            {
                if(i == j)
                {
                    
                    JLabel tmp = new JLabel(titles[i], JLabel.CENTER);
                    tmp.setBorder(BorderFactory.createLineBorder(Color.black));
                    panel.add(tmp);
                    continue;
                }

                Vec xAxis = data.get(j);

                ScatterPlot sp = new ScatterPlot(xAxis, yAxis);
                sp.setPadding(0);
                sp.setBorder(BorderFactory.createLineBorder(Color.black));
                panel.add(sp);

            }
        }

        this.setContentPane(panel);
    }




}
