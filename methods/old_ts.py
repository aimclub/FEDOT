from methods.gap_generator import generate_gaps

csv_file = 'data/Traffic.csv'
# 30%
generate_gaps(csv_file=csv_file,
              gap_dict={550: 150,
                        1000: 140,
                        1600: 360,
                        2500: 620,
                        4050: 420,
                        5400: 200},
              gap_value=-100.0,
              column_name='gap',
              vis=True)

generate_gaps(csv_file=csv_file,
              gap_dict={2500: 1500},
              gap_value=-100.0,
              column_name='gap_center',
              vis=True)
