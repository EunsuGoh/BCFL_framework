import os
import click

@click.group()
def cli():
    pass

@cli.command()
@click.option('--scenario', help='bcfl scenario, "crowdsource / consortium"')
def build_image(scenario):
    crowd_dockerfile = f"./docker/Dockerfile.{scenario}"
    os.system(f"docker build -f {crowd_dockerfile} -t {scenario} ./")



@cli.command()
def test():
    click.echo("dddd")

cli()