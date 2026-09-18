import argparse


def generate_identity_3DLUT(dim, output_file):
    step = 1.0 / (dim - 1)
    with open(output_file, 'w') as f:
        for k in range(dim):
            for j in range(dim):
                for i in range(dim):
                    f.write('{:.6f}  {:.6f}  {:.6f}\n'.format(
                        step * i, step * j, step * k))

def generate_identity_2DLUT(dim, output_file):
    step = 1.0 / (dim - 1)
    with open(output_file, 'w') as f:
        for j in range(dim):
            for i in range(dim):
                f.write('{:.6f}  {:.6f}\n'.format(
                    step * i, step * j))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dim', type=int, required=True)

    args = parser.parse_args()

    output_file = 'Identity_2DLUT{}.txt'.format(args.dim)
    generate_identity_2DLUT(args.dim, output_file)
